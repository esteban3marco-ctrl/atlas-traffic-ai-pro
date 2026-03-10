"""
ATLAS Pro - Generador de Reportes PDF
=======================================
Genera reportes profesionales de rendimiento del sistema.
Incluye: métricas horarias, alertas, escenarios, gráficos.

Uso:
    from reportes_pdf import generar_reporte_diario
    pdf_bytes = generar_reporte_diario(fecha, intersection_id, ...)
"""

import io
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger("ATLAS.Reportes")

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm, cm
    from reportlab.lib.colors import HexColor, black, white, grey
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, HRFlowable
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    REPORTLAB_DISPONIBLE = True
except ImportError:
    REPORTLAB_DISPONIBLE = False
    logger.warning("reportlab no disponible. Instalar con: pip install reportlab")


# Colores ATLAS
ATLAS_BLUE = HexColor("#3b82f6") if REPORTLAB_DISPONIBLE else None
ATLAS_DARK = HexColor("#0c1020") if REPORTLAB_DISPONIBLE else None
ATLAS_GREEN = HexColor("#10b981") if REPORTLAB_DISPONIBLE else None
ATLAS_AMBER = HexColor("#f59e0b") if REPORTLAB_DISPONIBLE else None
ATLAS_RED = HexColor("#ef4444") if REPORTLAB_DISPONIBLE else None
ATLAS_PURPLE = HexColor("#8b5cf6") if REPORTLAB_DISPONIBLE else None
ATLAS_GREY = HexColor("#6b7280") if REPORTLAB_DISPONIBLE else None
ATLAS_LIGHT = HexColor("#f3f4f6") if REPORTLAB_DISPONIBLE else None


def _get_styles():
    """Estilos personalizados para reportes ATLAS."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        'ATLASTitle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=ATLAS_BLUE,
        spaceAfter=6,
    ))

    styles.add(ParagraphStyle(
        'ATLASSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        textColor=ATLAS_GREY,
        spaceAfter=20,
    ))

    styles.add(ParagraphStyle(
        'ATLASHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=ATLAS_BLUE,
        spaceBefore=16,
        spaceAfter=8,
    ))

    styles.add(ParagraphStyle(
        'ATLASBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=black,
    ))

    styles.add(ParagraphStyle(
        'ATLASSmall',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        textColor=ATLAS_GREY,
    ))

    styles.add(ParagraphStyle(
        'ATLASMetricValue',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        textColor=ATLAS_BLUE,
        alignment=TA_CENTER,
    ))

    styles.add(ParagraphStyle(
        'ATLASMetricLabel',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        textColor=ATLAS_GREY,
        alignment=TA_CENTER,
    ))

    return styles


def _header_footer(canvas, doc):
    """Header y footer para cada página."""
    canvas.saveState()

    # Header line
    canvas.setStrokeColor(ATLAS_BLUE)
    canvas.setLineWidth(2)
    canvas.line(20 * mm, A4[1] - 15 * mm, A4[0] - 20 * mm, A4[1] - 15 * mm)

    # Header text
    canvas.setFont('Helvetica-Bold', 9)
    canvas.setFillColor(ATLAS_BLUE)
    canvas.drawString(20 * mm, A4[1] - 13 * mm, "ATLAS Pro")

    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(ATLAS_GREY)
    canvas.drawRightString(A4[0] - 20 * mm, A4[1] - 13 * mm, "Traffic Intelligence System")

    # Footer
    canvas.setFont('Helvetica', 7)
    canvas.setFillColor(ATLAS_GREY)
    canvas.drawString(20 * mm, 12 * mm,
                      f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    canvas.drawCentredString(A4[0] / 2, 12 * mm, "CONFIDENCIAL")
    canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, f"Pag. {doc.page}")

    canvas.restoreState()


def _build_kpi_table(stats: Dict, styles) -> Table:
    """Tabla de KPIs principales."""
    kpis = [
        ("Throughput", f"{stats.get('avg_throughput_2min', 0):.1f}", "veh/ciclo"),
        ("Espera Media", f"{stats.get('avg_wait_2min', 0):.1f}", "segundos"),
        ("Reward", f"{stats.get('avg_reward_2min', 0):.1f}", "reward/step"),
        ("Uptime", f"{stats.get('uptime_hours', 0):.1f}h", ""),
        ("Decisiones", f"{stats.get('total_decisions', 0):,}", "totales"),
        ("Alertas", f"{stats.get('alerts_unresolved', 0)}", "sin resolver"),
    ]

    header = [[Paragraph(k[0], styles['ATLASMetricLabel']) for k in kpis]]
    values = [[Paragraph(k[1], styles['ATLASMetricValue']) for k in kpis]]
    units = [[Paragraph(k[2], styles['ATLASSmall']) for k in kpis]]

    data = header + values + units
    col_width = (A4[0] - 40 * mm) / 6

    table = Table(data, colWidths=[col_width] * 6)
    table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, ATLAS_LIGHT),
        ('BACKGROUND', (0, 0), (-1, 0), ATLAS_LIGHT),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    return table


def _build_hourly_table(datos: List[Dict], styles) -> Table:
    """Tabla de datos horarios."""
    header = [
        Paragraph("<b>Hora</b>", styles['ATLASSmall']),
        Paragraph("<b>Reward</b>", styles['ATLASSmall']),
        Paragraph("<b>Velocidad</b>", styles['ATLASSmall']),
        Paragraph("<b>Cola</b>", styles['ATLASSmall']),
        Paragraph("<b>Espera</b>", styles['ATLASSmall']),
        Paragraph("<b>Vehiculos</b>", styles['ATLASSmall']),
        Paragraph("<b>Emerg.</b>", styles['ATLASSmall']),
    ]

    data = [header]
    for row in datos[:24]:
        hora = row.get("hora", "")
        if isinstance(hora, str) and "T" in hora:
            hora = hora.split("T")[1][:5]
        data.append([
            Paragraph(str(hora), styles['ATLASSmall']),
            Paragraph(f"{row.get('avg_reward', 0):.1f}", styles['ATLASSmall']),
            Paragraph(f"{row.get('avg_speed', 0):.1f}", styles['ATLASSmall']),
            Paragraph(f"{row.get('avg_queue', 0):.1f}", styles['ATLASSmall']),
            Paragraph(f"{row.get('avg_wait', 0):.1f}", styles['ATLASSmall']),
            Paragraph(f"{row.get('total_vehicles', 0)}", styles['ATLASSmall']),
            Paragraph(f"{row.get('emergencias', 0)}", styles['ATLASSmall']),
        ])

    col_widths = [25 * mm, 20 * mm, 22 * mm, 18 * mm, 20 * mm, 25 * mm, 18 * mm]
    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, ATLAS_LIGHT),
        ('BACKGROUND', (0, 0), (-1, 0), ATLAS_BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, ATLAS_LIGHT]),
    ]))
    return table


def _build_scenarios_table(escenarios: Dict, styles) -> Table:
    """Tabla de rendimiento por escenario."""
    header = [
        Paragraph("<b>Escenario</b>", styles['ATLASSmall']),
        Paragraph("<b>Reward</b>", styles['ATLASSmall']),
        Paragraph("<b>Ronda</b>", styles['ATLASSmall']),
        Paragraph("<b>Mejora vs Fijo</b>", styles['ATLASSmall']),
        Paragraph("<b>Episodios</b>", styles['ATLASSmall']),
    ]

    data = [header]
    for name, info in escenarios.items():
        data.append([
            Paragraph(name.capitalize(), styles['ATLASSmall']),
            Paragraph(f"{info.get('best_reward', 0):.1f}", styles['ATLASSmall']),
            Paragraph(str(info.get('ronda', '')), styles['ATLASSmall']),
            Paragraph(str(info.get('improvement_vs_fixed', '')), styles['ATLASSmall']),
            Paragraph(str(info.get('episodes_trained', '')), styles['ATLASSmall']),
        ])

    col_widths = [30 * mm, 22 * mm, 20 * mm, 30 * mm, 25 * mm]
    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, ATLAS_LIGHT),
        ('BACKGROUND', (0, 0), (-1, 0), ATLAS_PURPLE),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, ATLAS_LIGHT]),
    ]))
    return table


def _build_alerts_table(alertas: List[Dict], styles) -> Table:
    """Tabla de alertas recientes."""
    header = [
        Paragraph("<b>Hora</b>", styles['ATLASSmall']),
        Paragraph("<b>Severidad</b>", styles['ATLASSmall']),
        Paragraph("<b>Mensaje</b>", styles['ATLASSmall']),
        Paragraph("<b>Estado</b>", styles['ATLASSmall']),
    ]

    data = [header]
    for alert in alertas[:20]:
        ts = alert.get("timestamp", "")
        if isinstance(ts, str) and "T" in ts:
            ts = ts.split("T")[1][:8]
        sev = alert.get("severity", alert.get("severidad", "info"))
        data.append([
            Paragraph(str(ts), styles['ATLASSmall']),
            Paragraph(sev.upper(), styles['ATLASSmall']),
            Paragraph(str(alert.get("message", alert.get("descripcion", ""))), styles['ATLASSmall']),
            Paragraph("Resuelta" if alert.get("resolved", alert.get("resuelta")) else "Activa",
                       styles['ATLASSmall']),
        ])

    col_widths = [22 * mm, 22 * mm, 80 * mm, 22 * mm]
    table = Table(data, colWidths=col_widths)

    style_cmds = [
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, ATLAS_LIGHT),
        ('BACKGROUND', (0, 0), (-1, 0), ATLAS_AMBER),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]

    # Color por severidad
    for i, alert in enumerate(alertas[:20], start=1):
        sev = alert.get("severity", alert.get("severidad", "info")).lower()
        if sev == "critical" or sev == "emergency":
            style_cmds.append(('BACKGROUND', (1, i), (1, i), HexColor("#fee2e2")))
        elif sev == "warning":
            style_cmds.append(('BACKGROUND', (1, i), (1, i), HexColor("#fef3c7")))

    table.setStyle(TableStyle(style_cmds))
    return table


def generar_reporte_diario(
    fecha: str,
    intersection_id: str,
    datos_horarios: List[Dict],
    alertas: List[Dict],
    estadisticas: Dict,
    escenarios: Dict,
) -> bytes:
    """
    Genera un reporte PDF diario completo.
    Retorna bytes del PDF.
    """
    if not REPORTLAB_DISPONIBLE:
        raise ImportError("reportlab no disponible")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
    )

    styles = _get_styles()
    story = []

    # ---- PORTADA ----
    story.append(Spacer(1, 20 * mm))
    story.append(Paragraph("ATLAS Pro", styles['ATLASTitle']))
    story.append(Paragraph(
        f"Reporte Diario de Rendimiento — {fecha}<br/>"
        f"Interseccion: {intersection_id}",
        styles['ATLASSubtitle']
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=ATLAS_BLUE))
    story.append(Spacer(1, 10 * mm))

    # ---- KPIs ----
    story.append(Paragraph("Resumen de Indicadores", styles['ATLASHeading']))
    story.append(_build_kpi_table(estadisticas, styles))
    story.append(Spacer(1, 8 * mm))

    # ---- Rendimiento por Escenario ----
    story.append(Paragraph("Rendimiento por Escenario de Entrenamiento", styles['ATLASHeading']))
    story.append(Paragraph(
        "Comparativa del agente RL contra timing fijo en cada escenario de trafico.",
        styles['ATLASBody']
    ))
    story.append(Spacer(1, 4 * mm))
    story.append(_build_scenarios_table(escenarios, styles))
    story.append(Spacer(1, 8 * mm))

    # ---- Datos Horarios ----
    story.append(Paragraph("Metricas Horarias (Ultimas 24h)", styles['ATLASHeading']))
    story.append(_build_hourly_table(datos_horarios, styles))
    story.append(Spacer(1, 8 * mm))

    # ---- Alertas ----
    if alertas:
        story.append(Paragraph(f"Alertas Recientes ({len(alertas)})", styles['ATLASHeading']))
        story.append(_build_alerts_table(alertas, styles))
        story.append(Spacer(1, 8 * mm))

    # ---- Notas ----
    story.append(HRFlowable(width="100%", thickness=1, color=ATLAS_LIGHT))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "Este reporte ha sido generado automaticamente por ATLAS Pro v4.0. "
        "Los datos pueden provenir de simulacion o de mediciones reales "
        "dependiendo del modo de operacion del sistema.",
        styles['ATLASSmall']
    ))

    # Build PDF
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    logger.info(f"Reporte PDF generado: {len(pdf_bytes)} bytes, fecha={fecha}")
    return pdf_bytes


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import random

    print("Generando reporte de ejemplo...")

    # Datos de ejemplo
    from datetime import timedelta
    now = datetime.now()

    datos_horarios = []
    for i in range(24):
        h = (now - timedelta(hours=i)).hour
        factor = 1.8 if (7 <= h <= 9 or 17 <= h <= 19) else (0.3 if (h >= 22 or h <= 5) else 1.0)
        datos_horarios.append({
            "hora": (now - timedelta(hours=i)).isoformat(),
            "avg_reward": round(50 * factor + random.gauss(0, 10), 2),
            "avg_speed": round(35 / factor, 1),
            "avg_queue": round(8 * factor, 1),
            "avg_wait": round(20 * factor, 1),
            "total_vehicles": int(800 * factor),
            "emergencias": random.randint(0, 2) if factor > 1.5 else 0,
        })

    alertas = [
        {"timestamp": now.isoformat(), "severity": "warning", "message": "Cola excesiva en dir. E", "resolved": False},
        {"timestamp": now.isoformat(), "severity": "critical", "message": "MUSE: competencia baja", "resolved": False},
        {"timestamp": now.isoformat(), "severity": "info", "message": "OTA update disponible v1.2.3", "resolved": True},
    ]

    estadisticas = {
        "avg_throughput_2min": 42.3,
        "avg_wait_2min": 18.7,
        "avg_reward_2min": 67.4,
        "uptime_hours": 23.5,
        "total_decisions": 84600,
        "alerts_unresolved": 2,
    }

    escenarios = {
        "normal": {"best_reward": 45.2, "ronda": "R2", "improvement_vs_fixed": "+32%", "episodes_trained": 1100},
        "avenida": {"best_reward": 257.4, "ronda": "R4", "improvement_vs_fixed": "+41%", "episodes_trained": 1350},
        "evento": {"best_reward": 5.8, "ronda": "R4", "improvement_vs_fixed": "+18%", "episodes_trained": 1200},
        "heavy": {"best_reward": 118.1, "ronda": "R4", "improvement_vs_fixed": "+28%", "episodes_trained": 1150},
        "noche": {"best_reward": 773.5, "ronda": "R3", "improvement_vs_fixed": "+65%", "episodes_trained": 950},
        "emergencias": {"best_reward": 424.4, "ronda": "R4", "improvement_vs_fixed": "+55%", "episodes_trained": 1050},
    }

    pdf_data = generar_reporte_diario(
        fecha=now.strftime("%Y-%m-%d"),
        intersection_id="INT_001",
        datos_horarios=datos_horarios,
        alertas=alertas,
        estadisticas=estadisticas,
        escenarios=escenarios,
    )

    output_path = "atlas_report_example.pdf"
    with open(output_path, "wb") as f:
        f.write(pdf_data)
    print(f"Reporte guardado: {output_path} ({len(pdf_data):,} bytes)")
