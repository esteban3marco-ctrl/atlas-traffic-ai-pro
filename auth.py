"""
ATLAS Pro - Sistema de Autenticación JWT + Roles
==================================================
Módulo de autenticación para Dashboard v2:
- JWT tokens con expiración configurable
- 3 roles: admin, operador, visor
- Hashing seguro con bcrypt (fallback a hashlib)
- Middleware FastAPI para proteger endpoints
- Gestión de usuarios en PostgreSQL

Uso:
    from auth import AuthManager, get_current_user, require_role

    auth = AuthManager(secret_key="tu-clave-secreta")

    # En endpoints:
    @app.get("/api/protected")
    async def protected(user = Depends(get_current_user)):
        return {"user": user["username"], "role": user["role"]}

    @app.get("/api/admin-only")
    async def admin_only(user = Depends(require_role("admin"))):
        return {"admin": True}
"""

import os
import sys
import json
import time
import hashlib
import hmac
import base64
import secrets
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from functools import wraps

logger = logging.getLogger("ATLAS.Auth")

# ============================================================================
# JWT Implementation (sin dependencia externa)
# ============================================================================

class JWTManager:
    """
    Implementación JWT compacta sin dependencias externas.
    Usa HMAC-SHA256 para firmar tokens.
    """

    def __init__(self, secret_key: str, expiration_hours: int = 24):
        self.secret_key = secret_key.encode('utf-8')
        self.expiration_hours = expiration_hours

    def _base64url_encode(self, data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b'=').decode('utf-8')

    def _base64url_decode(self, data: str) -> bytes:
        padding = 4 - len(data) % 4
        if padding != 4:
            data += '=' * padding
        return base64.urlsafe_b64decode(data)

    def create_token(self, payload: Dict) -> str:
        """Crear JWT token"""
        header = {"alg": "HS256", "typ": "JWT"}

        # Añadir timestamps
        now = int(time.time())
        payload = {
            **payload,
            "iat": now,
            "exp": now + (self.expiration_hours * 3600),
        }

        # Encode header + payload
        header_b64 = self._base64url_encode(json.dumps(header).encode('utf-8'))
        payload_b64 = self._base64url_encode(json.dumps(payload).encode('utf-8'))

        # Sign
        message = f"{header_b64}.{payload_b64}".encode('utf-8')
        signature = hmac.new(self.secret_key, message, hashlib.sha256).digest()
        signature_b64 = self._base64url_encode(signature)

        return f"{header_b64}.{payload_b64}.{signature_b64}"

    def verify_token(self, token: str) -> Optional[Dict]:
        """Verificar y decodificar JWT token. Retorna payload o None."""
        try:
            parts = token.split('.')
            if len(parts) != 3:
                return None

            header_b64, payload_b64, signature_b64 = parts

            # Verificar firma
            message = f"{header_b64}.{payload_b64}".encode('utf-8')
            expected_sig = hmac.new(self.secret_key, message, hashlib.sha256).digest()
            actual_sig = self._base64url_decode(signature_b64)

            if not hmac.compare_digest(expected_sig, actual_sig):
                logger.warning("JWT: firma inválida")
                return None

            # Decodificar payload
            payload = json.loads(self._base64url_decode(payload_b64))

            # Verificar expiración
            if payload.get("exp", 0) < int(time.time()):
                logger.info("JWT: token expirado")
                return None

            return payload

        except Exception as e:
            logger.warning(f"JWT: error verificando token: {e}")
            return None


# ============================================================================
# Password Hashing
# ============================================================================

def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Hash password con SHA-256 + salt. Retorna (hash, salt)."""
    if salt is None:
        salt = secrets.token_hex(16)
    pw_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        iterations=100_000
    )
    return pw_hash.hex(), salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verificar password contra hash almacenado."""
    pw_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(pw_hash, stored_hash)


# ============================================================================
# Roles y Permisos
# ============================================================================

ROLES = {
    "admin": {
        "description": "Administrador del sistema",
        "permissions": [
            "view_dashboard", "view_metrics", "view_alerts", "view_history",
            "change_mode", "resolve_alerts", "manage_users", "view_audit",
            "generate_reports", "configure_system", "manual_control",
            "export_data", "manage_devices",
        ]
    },
    "operador": {
        "description": "Operador de tráfico",
        "permissions": [
            "view_dashboard", "view_metrics", "view_alerts", "view_history",
            "change_mode", "resolve_alerts", "manual_control",
            "generate_reports",
        ]
    },
    "visor": {
        "description": "Visualización de datos (solo lectura)",
        "permissions": [
            "view_dashboard", "view_metrics", "view_alerts", "view_history",
        ]
    },
}


def has_permission(role: str, permission: str) -> bool:
    """Verificar si un rol tiene un permiso específico."""
    return permission in ROLES.get(role, {}).get("permissions", [])


# ============================================================================
# User Store (PostgreSQL con fallback a archivo JSON)
# ============================================================================

class UserStore:
    """
    Almacén de usuarios. Usa PostgreSQL si está disponible,
    o un archivo JSON local como fallback para desarrollo.
    """

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url
        self.db_conn = None
        self._json_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config", "users.json"
        )
        self._users: Dict[str, Dict] = {}

        if db_url:
            try:
                import psycopg2
                self.db_conn = psycopg2.connect(db_url)
                self._init_db()
                logger.info("UserStore: conectado a PostgreSQL")
            except Exception as e:
                logger.warning(f"UserStore: PostgreSQL no disponible ({e}), usando JSON")
                self.db_conn = None

        if not self.db_conn:
            self._load_json()

    def _init_db(self):
        """Crear tabla de usuarios si no existe y poblar con defaults."""
        with self.db_conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS usuarios (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password_hash VARCHAR(128) NOT NULL,
                    salt VARCHAR(64) NOT NULL,
                    role VARCHAR(20) NOT NULL DEFAULT 'visor',
                    nombre VARCHAR(100),
                    email VARCHAR(100),
                    activo BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    last_login TIMESTAMPTZ
                );
            """)
            self.db_conn.commit()

            # Crear usuarios por defecto si la tabla está vacía
            cur.execute("SELECT COUNT(*) FROM usuarios")
            count = cur.fetchone()[0]
            if count == 0:
                logger.info("UserStore: tabla vacía, creando usuarios por defecto")
                self._create_default_users()

    def _load_json(self):
        """Cargar usuarios desde JSON."""
        if os.path.exists(self._json_path):
            try:
                with open(self._json_path, 'r') as f:
                    self._users = json.load(f)
                logger.info(f"UserStore: {len(self._users)} usuarios cargados desde JSON")
            except Exception:
                self._users = {}
        else:
            # Crear usuarios por defecto
            self._create_default_users()

    def _save_json(self):
        """Guardar usuarios a JSON."""
        os.makedirs(os.path.dirname(self._json_path), exist_ok=True)
        with open(self._json_path, 'w') as f:
            json.dump(self._users, f, indent=2, default=str)

    def _create_default_users(self):
        """Crear usuarios por defecto para desarrollo."""
        defaults = [
            ("admin", "atlas2026", "admin", "Administrador ATLAS", "admin@atlas.local"),
            ("operador", "trafico2026", "operador", "Operador de Tráfico", "operador@atlas.local"),
            ("visor", "visor2026", "visor", "Visor Dashboard", "visor@atlas.local"),
        ]
        for username, password, role, nombre, email in defaults:
            self.create_user(username, password, role, nombre, email)
        logger.info(f"UserStore: creados {len(defaults)} usuarios por defecto")

    def create_user(self, username: str, password: str, role: str = "visor",
                    nombre: str = "", email: str = "") -> bool:
        """Crear nuevo usuario."""
        if role not in ROLES:
            logger.error(f"Rol inválido: {role}")
            return False

        pw_hash, salt = hash_password(password)

        if self.db_conn:
            try:
                with self.db_conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO usuarios (username, password_hash, salt, role, nombre, email)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (username) DO NOTHING
                    """, (username, pw_hash, salt, role, nombre, email))
                    self.db_conn.commit()
                return True
            except Exception as e:
                logger.error(f"Error creando usuario en DB: {e}")
                self.db_conn.rollback()
                return False
        else:
            if username in self._users:
                return False
            self._users[username] = {
                "password_hash": pw_hash,
                "salt": salt,
                "role": role,
                "nombre": nombre,
                "email": email,
                "activo": True,
                "created_at": datetime.now().isoformat(),
                "last_login": None,
            }
            self._save_json()
            return True

    def authenticate(self, username: str, password: str) -> Optional[Dict]:
        """Autenticar usuario. Retorna datos del usuario o None."""
        if self.db_conn:
            try:
                with self.db_conn.cursor() as cur:
                    cur.execute("""
                        SELECT username, password_hash, salt, role, nombre, email, activo
                        FROM usuarios WHERE username = %s
                    """, (username,))
                    row = cur.fetchone()
                    if row and row[6]:  # activo
                        if verify_password(password, row[1], row[2]):
                            # Actualizar last_login
                            cur.execute(
                                "UPDATE usuarios SET last_login = NOW() WHERE username = %s",
                                (username,)
                            )
                            self.db_conn.commit()
                            return {
                                "username": row[0],
                                "role": row[3],
                                "nombre": row[4],
                                "email": row[5],
                            }
            except Exception as e:
                logger.error(f"Error autenticando en DB: {e}")
                return None
        else:
            user = self._users.get(username)
            if user and user.get("activo", True):
                if verify_password(password, user["password_hash"], user["salt"]):
                    user["last_login"] = datetime.now().isoformat()
                    self._save_json()
                    return {
                        "username": username,
                        "role": user["role"],
                        "nombre": user.get("nombre", ""),
                        "email": user.get("email", ""),
                    }
        return None

    def get_user(self, username: str) -> Optional[Dict]:
        """Obtener datos de usuario (sin password)."""
        if self.db_conn:
            try:
                with self.db_conn.cursor() as cur:
                    cur.execute("""
                        SELECT username, role, nombre, email, activo, created_at, last_login
                        FROM usuarios WHERE username = %s
                    """, (username,))
                    row = cur.fetchone()
                    if row:
                        return {
                            "username": row[0], "role": row[1],
                            "nombre": row[2], "email": row[3],
                            "activo": row[4], "created_at": str(row[5]),
                            "last_login": str(row[6]) if row[6] else None,
                        }
            except Exception:
                pass
        else:
            user = self._users.get(username)
            if user:
                return {
                    "username": username,
                    "role": user["role"],
                    "nombre": user.get("nombre", ""),
                    "email": user.get("email", ""),
                    "activo": user.get("activo", True),
                    "created_at": user.get("created_at"),
                    "last_login": user.get("last_login"),
                }
        return None

    def list_users(self) -> List[Dict]:
        """Listar todos los usuarios (sin passwords)."""
        if self.db_conn:
            try:
                with self.db_conn.cursor() as cur:
                    cur.execute("""
                        SELECT username, role, nombre, email, activo, created_at, last_login
                        FROM usuarios ORDER BY created_at
                    """)
                    return [
                        {
                            "username": r[0], "role": r[1], "nombre": r[2],
                            "email": r[3], "activo": r[4],
                            "created_at": str(r[5]),
                            "last_login": str(r[6]) if r[6] else None,
                        }
                        for r in cur.fetchall()
                    ]
            except Exception:
                return []
        else:
            return [
                {
                    "username": u,
                    "role": d["role"],
                    "nombre": d.get("nombre", ""),
                    "email": d.get("email", ""),
                    "activo": d.get("activo", True),
                    "created_at": d.get("created_at"),
                    "last_login": d.get("last_login"),
                }
                for u, d in self._users.items()
            ]

    def update_user(self, username: str, role: Optional[str] = None,
                    activo: Optional[bool] = None, nombre: Optional[str] = None) -> bool:
        """Actualizar datos de usuario."""
        if self.db_conn:
            try:
                updates = []
                params = []
                if role is not None and role in ROLES:
                    updates.append("role = %s")
                    params.append(role)
                if activo is not None:
                    updates.append("activo = %s")
                    params.append(activo)
                if nombre is not None:
                    updates.append("nombre = %s")
                    params.append(nombre)
                if not updates:
                    return False
                params.append(username)
                with self.db_conn.cursor() as cur:
                    cur.execute(
                        f"UPDATE usuarios SET {', '.join(updates)} WHERE username = %s",
                        params
                    )
                    self.db_conn.commit()
                return True
            except Exception:
                self.db_conn.rollback()
                return False
        else:
            user = self._users.get(username)
            if not user:
                return False
            if role is not None and role in ROLES:
                user["role"] = role
            if activo is not None:
                user["activo"] = activo
            if nombre is not None:
                user["nombre"] = nombre
            self._save_json()
            return True

    def change_password(self, username: str, new_password: str) -> bool:
        """Cambiar password de usuario."""
        pw_hash, salt = hash_password(new_password)
        if self.db_conn:
            try:
                with self.db_conn.cursor() as cur:
                    cur.execute(
                        "UPDATE usuarios SET password_hash = %s, salt = %s WHERE username = %s",
                        (pw_hash, salt, username)
                    )
                    self.db_conn.commit()
                return True
            except Exception:
                self.db_conn.rollback()
                return False
        else:
            if username not in self._users:
                return False
            self._users[username]["password_hash"] = pw_hash
            self._users[username]["salt"] = salt
            self._save_json()
            return True


# ============================================================================
# AuthManager (integra JWT + UserStore)
# ============================================================================

class AuthManager:
    """
    Manager principal de autenticación.
    Integra JWT, UserStore y permisos.
    """

    def __init__(self, secret_key: Optional[str] = None, db_url: Optional[str] = None,
                 expiration_hours: int = 24):
        if secret_key is None:
            secret_key = os.environ.get("ATLAS_JWT_SECRET", "atlas-pro-jwt-secret-2026")
        self.jwt = JWTManager(secret_key, expiration_hours)
        self.user_store = UserStore(db_url)
        logger.info("AuthManager inicializado")

    def login(self, username: str, password: str) -> Optional[Dict]:
        """
        Autenticar usuario y generar token JWT.
        Retorna {"token": "...", "user": {...}} o None.
        """
        user = self.user_store.authenticate(username, password)
        if user is None:
            logger.warning(f"Login fallido para: {username}")
            return None

        token = self.jwt.create_token({
            "sub": user["username"],
            "role": user["role"],
            "nombre": user.get("nombre", ""),
        })

        logger.info(f"Login exitoso: {username} (rol: {user['role']})")
        return {
            "token": token,
            "token_type": "bearer",
            "expires_in": self.jwt.expiration_hours * 3600,
            "user": user,
        }

    def verify(self, token: str) -> Optional[Dict]:
        """Verificar token y retornar datos del usuario."""
        payload = self.jwt.verify_token(token)
        if payload is None:
            return None
        return {
            "username": payload.get("sub"),
            "role": payload.get("role"),
            "nombre": payload.get("nombre", ""),
        }

    def check_permission(self, token: str, permission: str) -> bool:
        """Verificar si el token tiene un permiso específico."""
        user = self.verify(token)
        if user is None:
            return False
        return has_permission(user["role"], permission)


# ============================================================================
# FastAPI Dependencies
# ============================================================================

# Singleton global (se inicializa en api_produccion.py)
_auth_manager: Optional[AuthManager] = None


def init_auth(secret_key: Optional[str] = None, db_url: Optional[str] = None) -> AuthManager:
    """Inicializar el AuthManager global."""
    global _auth_manager
    _auth_manager = AuthManager(secret_key=secret_key, db_url=db_url)
    return _auth_manager


def get_auth() -> AuthManager:
    """Obtener el AuthManager global."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager


async def get_current_user(authorization: str = None) -> Dict:
    """
    FastAPI dependency: extraer usuario del header Authorization.
    Uso: user = Depends(get_current_user)
    """
    from fastapi import Header, HTTPException

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Token de autenticación requerido",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Extraer token del header "Bearer <token>"
    token = authorization
    if authorization.startswith("Bearer "):
        token = authorization[7:]

    auth = get_auth()
    user = auth.verify(token)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Token inválido o expirado",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_role(*roles: str):
    """
    FastAPI dependency factory: requerir uno de los roles especificados.
    Uso: user = Depends(require_role("admin", "operador"))
    """
    async def role_checker(authorization: str = None) -> Dict:
        from fastapi import HTTPException

        user = await get_current_user(authorization)
        if user["role"] not in roles:
            raise HTTPException(
                status_code=403,
                detail=f"Acceso denegado. Se requiere rol: {', '.join(roles)}",
            )
        return user
    return role_checker


# ============================================================================
# CLI para testing
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    auth = AuthManager()
    print("\n=== ATLAS Pro Auth System ===\n")

    # Listar usuarios
    users = auth.user_store.list_users()
    print(f"Usuarios ({len(users)}):")
    for u in users:
        print(f"  - {u['username']:12s} | {u['role']:10s} | {u['nombre']}")

    # Test login
    print("\n--- Test Login ---")
    result = auth.login("admin", "atlas2026")
    if result:
        print(f"Token: {result['token'][:50]}...")
        print(f"User: {result['user']}")

        # Verificar token
        verified = auth.verify(result["token"])
        print(f"Verified: {verified}")

        # Check permisos
        print(f"Can manage_users: {auth.check_permission(result['token'], 'manage_users')}")
        print(f"Can view_dashboard: {auth.check_permission(result['token'], 'view_dashboard')}")

    # Test login visor
    result_visor = auth.login("visor", "visor2026")
    if result_visor:
        print(f"\nVisor can manage_users: {auth.check_permission(result_visor['token'], 'manage_users')}")
        print(f"Visor can view_dashboard: {auth.check_permission(result_visor['token'], 'view_dashboard')}")

    print("\n=== Test completado ===")
