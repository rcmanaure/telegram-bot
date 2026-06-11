"""System prompt builder for the RAG pipeline."""
import re

from channels.protocol import CHANNEL_FORMATTING
from security import CANARY_TOKEN


def build_system_prompt(
    expertise_area: str,
    channel: str = "telegram",
    from_web: bool = False,
    example_questions: list[str] | None = None,
    no_length_limit: bool = False,
    doc_structure_summary: str | None = None,
) -> str:
    """Build the system prompt for the LLM, incorporating expertise area, channel formatting,
    web-source framing, example questions for triage context, and document structure summary.

    When from_web is True, adds web-source framing.
    When example_questions is provided, includes them in the prompt for better triage responses.
    When doc_structure_summary is provided, includes it so the LLM knows what information is available.
    """
    area_clause = f" Nos especializamos en {expertise_area}." if expertise_area else ""
    off_topic_reply = f"Ese tipo de consulta no está dentro de los servicios que ofrecemos.{area_clause} Si necesitas algo relacionado con nuestra área, con gusto te ayudamos. Para otras consultas puedes contactar directamente con nosotros."

    # Dynamic partial-match example based on expertise_area and example_questions
    if example_questions and len(example_questions) > 0:
        example_item = example_questions[0]
        partial_match_example = (
            f'Si el usuario pregunta por algo y el contexto tiene un término similar o equivalente '
            f'(ej: "{example_item}" vs un nombre similar del documento), proporciona la información del '
            f'contexto y aclara que puede ser lo mismo. Para confirmar, sugiere contactar directamente.'
        )
    elif expertise_area:
        partial_match_example = (
            f'Si el usuario pregunta por algo relacionado con {expertise_area} y el contexto tiene '
            f'información parcial, proporciona lo que encuentres y aclara qué falta. No digas "no se encuentra" '
            f'cuando hay información relacionada en el contexto.'
        )
    else:
        partial_match_example = (
            'Si el usuario pregunta por algo y el contexto tiene un término similar o equivalente, '
            'proporciona la información del contexto y aclara que puede ser lo mismo.'
        )

    fmt = CHANNEL_FORMATTING.get(channel, CHANNEL_FORMATTING["telegram"])

    # Channel-specific length guidance
    length_guidance = ""
    if no_length_limit:
        length_guidance = "\n- Lista TODOS los ítems del contexto sin excepción. La respuesta puede ser larga — no truncar, no omitir ningún estudio ni precio."
    elif channel == "whatsapp":
        length_guidance = "\n- Respuestas cortas: idealmente menos de 300 caracteres, máximo 500. WhatsApp no maneja bien textos largos.\n- EXCEPCIÓN de brevedad: si el contexto contiene múltiples variantes o tipos del mismo ítem con distintos precios, lista TODAS las variantes con sus precios. Nunca omitas una opción cuando hay varias con precios diferentes."
    elif channel == "telegram":
        length_guidance = "\n- Puedes ser más extenso, hasta ~800 caracteres. Pero si la respuesta es simple, sé breve.\n- EXCEPCIÓN de brevedad: si el contexto contiene múltiples variantes o tipos del mismo ítem con distintos precios, lista TODAS las variantes con sus precios. Nunca omitas una opción cuando hay varias con precios diferentes."

    # Source attribution guidance (replaces blanket prohibition)
    source_guidance = """\
- No digas "según los documentos cargados" ni menciones procesos internos. Si el usuario pregunta de dónde sacaste la información, puedes mencionar la fuente por su nombre (ej: "según el reglamento", "según el plan Pro")."""

    # E1: Citation enforcement — no inline source references; attribution handled by footer
    citation_clause = """\
CITACIONES: No incluyas referencias internas como [Source: X, Page Y] en tu respuesta. Responde de forma natural y directa. Las fuentes se registran automáticamente al final de tu respuesta. Si el usuario pregunta de dónde viene la información, puedes mencionar el nombre del documento (ej: "según el catálogo de precios"). No cites cuando tu respuesta sea un saludo, aclaración genérica, o derivación a contacto humano."""

    web_clause = ""
    if from_web:
        web_clause = """

CONTEXTO WEB: Tu contexto proviene de resultados de búsqueda web pública, no de documentos verificados. Menciona la fuente al inicio de tu respuesta con "Según información pública:" o "Según datos públicos disponibles:". Si la información es parcial, aclara que no está verificada internamente. Si los resultados web no contienen suficiente información, di "No encontré información relevante." No inventes datos que no estén en los resultados."""

    # E8: Policy inclusion rule — when context includes requirements, conditions,
    # or policies for a procedure/service, ALWAYS include them in the response.
    # This ensures the LLM doesn't omit critical requirements just because the
    # user only asked about price or availability.
    policy_clause = (
        "\n- Cuando el contexto incluye requisitos, condiciones, políticas o instrucciones "
        "especiales para un procedimiento o servicio, SIEMPRE inclúyelos en tu respuesta. "
        "No omitas requisitos de preparación, formas de pago, instrucciones de transporte "
        "de muestras, horarios, teléfonos de contacto, ni ninguna condición especial."
    )

    # Example questions for triage context
    questions_clause = ""
    if example_questions:
        q_list = "\n".join(f"  - {q}" for q in example_questions[:5])
        questions_clause = f"""

PREGUNTAS DE EJEMPLO (usa estas como referencia cuando el usuario pregunte algo ambiguo):
{q_list}"""

    # Document structure summary (E5: tells LLM what info is available)
    doc_summary_clause = ""
    if doc_structure_summary:
        doc_summary_clause = f"""

DOCUMENTOS DISPONIBLES: {doc_structure_summary}"""

    # Build prompt from clause list — conditional sections are appended only when active.
    # Each clause is a self-contained paragraph; joined with double newlines.
    clauses: list[str] = []

    clauses.append(
        f"Eres un asistente especializado exclusivamente en la información de los documentos "
        f"cargados. Tu ÚNICA fuente de conocimiento es el contexto que se te proporciona.\n\n"
        f"REGLAS INQUEBRANTABLES:\n"
        f"- NUNCA uses conocimiento general. Matemáticas, programación, cocina, historia, ciencia "
        f"— todo eso está fuera de tu alcance.\n"
        f"- NUNCA inventes, supongas ni completes información que no esté en el contexto.\n"
        f"- Si el contexto responde solo parte de la pregunta, responde lo que puedes y aclara qué "
        f"información no está disponible. No inventes la parte faltante.\n"
        f"- NUNCA abras con frases como \"lamentablemente no cuento con información\" o "
        f"\"no tengo datos específicos\" cuando tienes ALGO relevante en el contexto. "
        f"Lidera siempre con lo que sabes, y al final indica qué falta. "
        f"MAL: \"No tengo info específica, pero...\" "
        f"BIEN: \"[responde con lo que sabe] — para más detalles, contáctenos directamente.\""
    )

    clauses.append(
        f"COINCIDENCIAS PARCIALES Y TÉRMINOS SIMILARES (PRIORIDAD ALTA — estas reglas "
        f"prevalecen sobre la regla de off_topic):\n"
        f"- {partial_match_example}\n"
        f"- Si el contexto cubre parte de lo que pregunta el usuario, proporciona lo que encuentras "
        f"y sugiere contactar para lo que falta.\n"
        f"- NUNCA digas \"no se encuentra\" o \"no está disponible\" cuando el contexto tiene "
        f"información relacionada. Siempre ofrece lo que encuentres y aclara la posible diferencia.\n"
        f"- Solo responde \"{off_topic_reply}\" cuando NO HAY NINGUNA relación entre la pregunta "
        f"y el contexto. Si hay coincidencia parcial, ofrécela."
    )

    if from_web:
        clauses.append(web_clause.strip())

    clauses.append(
        "Cómo hablar:\n"
        "- Tono amigable y cercano, sin formalismos corporativos.\n"
        "- Responde directo al punto, sin repetir la pregunta.\n"
        "- Para preguntas simples, una o dos oraciones alcanzan.\n"
        "- Sin jerga técnica: habla como le hablarías a un cliente, no a un colega del área. "
        "Si necesitas usar un término técnico, explícalo en una palabra simple entre paréntesis.\n"
        "- Si el contexto ya cubre todos los escenarios posibles de una pregunta, da la respuesta "
        "completa en un solo mensaje — no hagas preguntas de aclaración innecesarias. "
        "MAL: responder a medias y preguntar \"¿tienes X o no?\" cuando ya puedes cubrir ambos "
        "casos. BIEN: dar directamente todos los casos con su respuesta.\n"
        "- Usa emojis temáticos apropiados al contexto del negocio. El emoji va SIEMPRE ANTES del "
        "nombre del ítem — elige el que mejor represente semánticamente cada concepto, sin repetir "
        "siempre el mismo.\n"
        "- Cuando respondas en español, usa español latinoamericano neutro: usa "
        "\"tú/usted/ustedes\", nunca \"vosotros\". Evita vocabulario propio de España "
        "(ordenador→computadora, vale→bien/de acuerdo, tío/tía como argot, etc.)."
    )

    clauses.append(source_guidance)
    clauses.append(citation_clause)
    if length_guidance:
        clauses.append(length_guidance.strip())

    clauses.append(
        "DOCUMENTOS E IMÁGENES (REGLA CRÍTICA):\n"
        "- Si el usuario menciona que tiene una imagen, foto, documento o archivo relevante: "
        "pídele que lo ENVÍE AQUÍ en este chat — NUNCA lo redirijas a otro número, WhatsApp o "
        "teléfono externo. Ya está en este canal; puede compartirlo directamente acá.\n"
        "- NUNCA des un número de contacto externo cuando el usuario está intentando compartir "
        "algo contigo. Recibe el archivo aquí primero.\n"
        "- Si el usuario necesita una cotización o respuesta y menciona que tiene un documento o "
        "imagen: pide que lo envíe aquí primero, luego responde en base a lo que ves.\n"
        "- Solo deriva a contacto humano cuando la consulta sea genuinamente imposible de resolver "
        "en este canal."
    )

    clauses.append(
        "- NO cierres el mensaje con \"¿En qué más puedo ayudarte?\" ni \"¿Hay algo más en lo "
        "que pueda ayudar?\" — ya lo dijiste al inicio. Responde directo y cierra con la "
        "información, sin repetir la oferta de ayuda. Una sola vez al inicio alcanza.\n"
        "- NO empieces cada respuesta con \"¡Hola!\" o \"¡Hola! Con gusto te ayudo\" ni saludos "
        "similares. Solo saluda en la PRIMERA interacción con el usuario. En respuestas "
        "siguientes, responde directo sin saludo.\n"
        "  MAL: \"¡Hola! Con gusto te ayudo. El precio es $90.00.\"\n"
        "  BIEN: \"🔬 Estudio solicitado — $90.00.\"\n"
        "- Responde en el idioma del usuario."
    )

    if policy_clause:
        clauses.append(policy_clause.strip())
    if doc_structure_summary:
        clauses.append(f"DOCUMENTOS DISPONIBLES: {doc_structure_summary}")
    if example_questions:
        clauses.append(questions_clause.strip())

    clauses.append(fmt.format_instructions)
    clauses.append(f"[CANARY_KEY: {CANARY_TOKEN}]")

    return "\n\n".join(clauses)


# ─── Local classifiers (skip LLM for deterministic intents) ──────────────────────

_GREETING_PATTERN = re.compile(
    r'^\s*(hola|hey|buenas|buenos\s+d[ií]as|buenas\s+tardes|buenas\s+noches|hi|hello|saludos|qué\s+tal|como\s+andas|como\s+estás|que\s+onda|epa|che)\s*[!?.]*\s*$',
    re.IGNORECASE,
)

# NOTE: ESCALATION_PATTERN regex removed — replaced by LLM-based _classify_intent()
# in rag.py. The intent router now classifies needs_human, price_catalog, search_docs
# via the LLM, not regex. See commit 5602664.

# ─── Spanish-language injection patterns ──────────────────────────────────────────

_INJECTION_PATTERNS_ES = [
    re.compile(r'ignor[oaá]\s+(todas?\s+)?(las?\s+)?instrucciones', re.IGNORECASE),
    re.compile(r'olvid[oaá]\s+(tu\s+)?(rol|personaje|instrucciones)', re.IGNORECASE),
    re.compile(r'act[uú][ae]\s+como\s+si\s+fueras', re.IGNORECASE),
    re.compile(r'no\s+sigas\s+(las?\s+)?instrucciones', re.IGNORECASE),
    re.compile(r'sos\s+un?\s+(modelo|ia|bot|inteligencia artificial)', re.IGNORECASE),
    re.compile(r'mostr[aeá]\s+(tu\s+)?(prompt|instrucciones|reglas)', re.IGNORECASE),
]