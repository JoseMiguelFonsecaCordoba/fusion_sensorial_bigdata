"""Léxico de dimensiones turísticas y mapeo sensorial -> turístico.

El texto del dataset se normaliza sin acentos y en minúsculas antes de buscar,
por lo que todas las palabras aquí van sin acentos. Cada dimensión es una
alternativa regex; una reseña "menciona" la dimensión si contiene alguna.
Se validó la cobertura sobre el corpus real (todas aparecen en >4% de reseñas).
"""

DIMS = {
    "vida_nocturna": r"bar|bares|antro|antros|fiesta|fiestas|dj|cocteles|coctel|cerveza|cervezas|tragos|mezcal|discoteca|bailar|baile|nocturna|musica vivo|musica en vivo|karaoke",
    "aventura": r"aventura|aventuras|tirolesa|tirolesas|rapel|kayak|buceo|snorkel|snorkeling|senderismo|caminata|caminatas|escalada|cuatrimoto|cuatrimotos|adrenalina|bicicleta|bicicletas|cabalgata|cabalgatas|nadar|surf|aventurero",
    "naturaleza": r"naturaleza|cenote|cenotes|cascada|cascadas|bosque|rio|laguna|lago|selva|montana|montanas|volcan|reserva|flora|fauna|manglar|manglares|ecologico|paisaje|paisajes|senderos|aves|jardin",
    "descanso": r"relajante|relajar|relajarse|tranquilo|tranquila|tranquilidad|spa|masaje|masajes|descansar|descanso|paz|silencio|sereno|temazcal|bienestar|yoga|calma|relax",
    "cultura": r"historia|historico|historica|museo|museos|arqueologica|arqueologico|piramide|piramides|ruinas|templo|iglesia|catedral|convento|artesanias|artesania|cultura|cultural|tradicion|tradiciones|arquitectura|colonial|maya|mayas|leyenda|plaza|centro historico",
    "gastronomia": r"platillos|platillo|sabor|sabores|deliciosa|delicioso|cocina|chef|menu|tacos|mariscos|postres|desayuno|gastronomia|gastronomica|antojitos|ceviche|cafe|pizza|sopa",
    "playa": r"playa|playas|mar|arena|oceano|isla|olas|costa|bahia|caribe|arrecife|catamaran|lancha",
    "valor_barato": r"barato|barata|baratos|baratas|economico|economica|economicos|economicas|accesible|accesibles|gratis|gratuito|gratuita|precio justo|buen precio|excelente precio|buenos precios|calidad precio|porciones grandes|porciones generosas|porcion generosa|abundante|abundantes|mochilero|mochileros|estudiante|estudiantes|presupuesto",
    "valor_caro": r"caro|cara|caros|caras|costoso|costosa|costosos|carisimo|carisima|sobrepreciado|elevado|elevados|excesivo|excesivos|trampa turistica|abusivo|abusivos|cobran de mas|robo",
}

TOURISM_DIMS = ["vida_nocturna", "aventura", "naturaleza", "descanso", "cultura", "gastronomia", "playa"]

LABELS = {
    "vida_nocturna": "vida nocturna", "aventura": "aventura", "naturaleza": "naturaleza",
    "descanso": "descanso y bienestar", "cultura": "cultura e historia",
    "gastronomia": "gastronomía", "playa": "playa y mar",
}

# Matriz M (hipótesis heurística, declarada como limitación): cada fila da el peso de
# los ejes sensoriales centrados (a, p, w en [-1, 1]) sobre una dimensión turística.
#              a (activación)  p (positividad)  w (calidez)
M = {
    "vida_nocturna": (0.7,  0.3,  0.0),
    "aventura":      (0.7,  0.1,  0.2),
    "naturaleza":    (-0.4, 0.0, -0.6),
    "descanso":      (-0.8, 0.0, -0.2),
    "cultura":       (0.0,  0.1,  0.6),
    "gastronomia":   (0.1,  0.3,  0.6),
    "playa":         (0.0,  0.6,  0.2),
}
