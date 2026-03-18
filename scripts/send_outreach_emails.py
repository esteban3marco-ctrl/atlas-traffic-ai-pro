import os
import re
import smtplib
import argparse
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

def parse_municipalities(file_path):
    """
    Parsea las tablas de municipios del archivo markdown.
    Captura: Ciudad, Población, Foco Comercial, Contacto / Área
    """
    municipalities = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Expresión regular para capturar filas de tablas con el formato:
    # | **Nombre** | Población | Foco Comercial | Contacto / Área |
    table_rows = re.findall(r'\| \*\*([^*]+)\*\* \| ([^|]+) \| ([^|]+) \| ([^|]+) \|', content)
    
    for name, pop, focus, contact in table_rows:
        contact = contact.strip()
        name = name.strip()
        focus = focus.strip()
        
        # Omitir cabeceras o filas vacías
        if name.lower() in ["ciudad", ":---"]:
            continue
            
        # Verificar si hay un email o un área de contacto
        email = None
        if '@' in contact:
            email = contact
        
        municipalities.append({
            'name': name,
            'population': pop.strip(),
            'focus': focus,
            'contact_raw': contact,
            'email': email
        })
    
    return municipalities

def generate_body(municipality, sender_config, custom_text=None):
    """
    Genera el cuerpo del mensaje personalizado.
    """
    if custom_text:
        return custom_text

    m_name = municipality['name']
    focus = municipality['focus']
    
    # Personalización basada en el foco comercial
    placeholder_text = f"el reto de la movilidad en {m_name}"
    if focus and focus != "N/A":
        placeholder_text = f"vuestro enfoque en {focus}"

    body = f"""Estimado/a Responsable de Movilidad:

He estado analizando la infraestructura semafórica de {m_name} y creo firmemente que tenéis el perfil perfecto para dar el siguiente paso hacia una Smart City real.

Teniendo en cuenta que uno de los retos locales suele ser {placeholder_text}, hemos desarrollado el sistema operativo ATLAS Pro para transformar el control de tráfico y dotarlo de Inteligencia Distribuida. Sus ventajas principales:

1. Incremento del 30% en fluidez: La IA aprende y se adapta al milisegundo al flujo real, minimizando tiempos muertos y generando 'Ondas Verdes' dinámicas.
2. Cumplimiento ZBE (Cero Emisiones): Al reducir radicalmente la dinámica "arranca-para", los niveles de emisiones contaminantes bajan drásticamente allí donde más se necesita.
3. Instalación 'Invisible': Se integra por software directamente en las cámaras y controladores que ya tenéis. Cero obras y un ROI inmediato (hasta 90% más económico que licencias legacy).
4. Resiliencia ante Atascos y Accidentes: Identifica bloqueos mediante visión y redirige automáticamente las arterias adyacentes para evitar colapsos irreversibles.

Tenemos datos recientes y estudios de caso que demuestran la eficacia de esta tecnología en redes de tamaño y problemática comparable.

Si consideras que este enfoque podría encajar en los planes de innovación técnica de vuestra concejalía, respóndeme a este correo y estaré encantado de enviarte datos visuales o detallarte más sobre el funcionamiento del sistema.

Atentamente,

{sender_config['name']}
{sender_config['position']}"""
    
    return body

def send_email(municipality, smtp_config, sender_config, body=None, dry_run=True):
    """
    Envía el correo personalizado.
    """
    m_name = municipality['name']
    m_email = municipality['email']
    
    if not m_email:
        print(f"[!] No hay email configurado para {m_name}. Saltando...")
        return False
        
    subject = f"[Propuesta] Reducción del 30% en atascos y emisiones para {m_name}"
    
    if body is None:
        body = generate_body(municipality, sender_config)

    if dry_run:
        print(f"\n--- [VISTA PREVIA] Correo para {m_name} ({m_email}) ---")
        print(f"Asunto: {subject}")
        print("-" * 20)
        print(body)
        print("-" * 42 + "\n")
        return True

    from email.message import EmailMessage
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = f"{sender_config['name']} <{sender_config['email']}>"
    msg['To'] = m_email
    msg.set_content(body)

    try:
        server = smtplib.SMTP(smtp_config['server'], smtp_config['port'])
        server.starttls()
        server.login(smtp_config['email'], smtp_config['password'].replace(' ', ''))
        server.send_message(msg)
        server.quit()
        print(f"[OK] Correo enviado con éxito a {m_name} ({m_email})")
        return True
    except Exception as e:
        print(f"[ERROR] Error enviando a {m_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Automatización de correos de outreach para ATLAS Pro")
    parser.add_argument("--send", action="store_true", help="Si se especifica, envía los correos realmente")
    parser.add_argument("--city", type=str, help="Enviar solo a una ciudad específica (nombre exacto)")
    parser.add_argument("--draft-file", type=str, help="Ruta a un archivo .txt con un mensaje personalizado")
    args = parser.parse_args()

    doc_path = os.path.join("docs", "ESTRATEGIA_OUTREACH_MUNICIPIOS.md")
    municipalities = parse_municipalities(doc_path)
    
    if args.city:
        municipalities = [m for m in municipalities if m['name'].lower() == args.city.lower()]
        if not municipalities:
            print(f"No se encontró la ciudad '{args.city}' en el documento.")
            return

    # Configuración desde .env
    smtp_config = {
        'server': os.getenv('SMTP_SERVER'),
        'port': int(os.getenv('SMTP_PORT', 587)),
        'email': os.getenv('SMTP_EMAIL'),
        'password': os.getenv('SMTP_PASSWORD')
    }
    
    sender_config = {
        'name': os.getenv('SENDER_NAME', 'Esteban Marco'),
        'position': os.getenv('SENDER_POSITION', 'CEO & Founder, ATLAS AI'),
        'email': os.getenv('SMTP_EMAIL')
    }

    custom_body = None
    if args.draft_file:
        try:
            with open(args.draft_file, 'r', encoding='utf-8') as df:
                custom_body = df.read()
        except Exception as e:
            print(f"Error leyendo el archivo de borrador: {e}")
            return

    for m in municipalities:
        if m['name'] in ["Granada", "Almería", "Burgos", "Pozuelo de Alarcón", "Las Rozas"]:
            print(f"[!] {m['name']} ya fue contactado hoy. Saltando...")
            continue
        body = generate_body(m, sender_config, custom_text=custom_body)
        if args.send:
            if not m.get('email'):
                print(f"[!] No hay email configurado para {m['name']}. Saltando...")
                continue
            
            print(f"\nGenerando propuesta para {m['name']} ({m['email']})...")
            print("-" * 20)
            print(body)
            print("-" * 20)
            print(f"Enviando correo a {m['email']}...")
            send_email(m, smtp_config, sender_config, body=body, dry_run=False)
        else:
            send_email(m, smtp_config, sender_config, body=body, dry_run=True)

if __name__ == "__main__":
    main()
