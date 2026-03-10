"""Scan all SUMO scenario files for edge/TL info."""
import xml.etree.ElementTree as ET
import os

scenarios = [
    'simple', 'simple_hora_punta', 'simple_noche', 'simple_emergencias',
    'hora_punta', 'noche', 'complejo', 'cruce_t', 'doble',
    'avenida', 'emergencias', 'evento'
]

for s in scenarios:
    net_file = f'simulations/{s}/intersection.net.xml'
    if not os.path.exists(net_file):
        print(f'{s}: NO NET FILE')
        continue
    tree = ET.parse(net_file)
    root = tree.getroot()
    edges = [e.get('id') for e in root.findall('.//edge') if not e.get('id', '').startswith(':')]
    tls = [t.get('id') for t in root.findall('.//tlLogic')]
    in_edges = [e for e in edges if '_in' in e or 'to' in e.lower()]
    print(f'{s}: in_edges={in_edges[:6]}, tls={tls}')
