from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent

try:
    import anthropic
except ImportError:  # pragma: no cover - depende del entorno
    anthropic = None


@dataclass
class ProbabilitySnapshot:
    home: float
    draw: float
    away: float

    def as_dict(self) -> dict[str, float]:
        return {"home": self.home, "draw": self.draw, "away": self.away}

    def top_outcome(self) -> tuple[str, float]:
        values = self.as_dict()
        winner = max(values, key=values.get)
        return winner, values[winner]


class BaseAnalyst:
    provider_name = "base"

    def analyze_probability_divergence(
        self,
        match_name: str,
        bookmaker: ProbabilitySnapshot,
        model_probs: ProbabilitySnapshot,
        polymarket: ProbabilitySnapshot | None = None,
    ) -> str:
        raise NotImplementedError


class ClaudeAnalyst(BaseAnalyst):
    provider_name = "claude"

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514") -> None:
        if anthropic is None:
            raise RuntimeError("El paquete anthropic no esta instalado.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def analyze_probability_divergence(
        self,
        match_name: str,
        bookmaker: ProbabilitySnapshot,
        model_probs: ProbabilitySnapshot,
        polymarket: ProbabilitySnapshot | None = None,
    ) -> str:
        polymarket_text = (
            f"- Polymarket: home={polymarket.home:.3f}, draw={polymarket.draw:.3f}, away={polymarket.away:.3f}\n"
            if polymarket
            else "- Polymarket: no disponible\n"
        )

        prompt = dedent(
            f"""
            Analiza estas probabilidades para un partido de futbol y escribe una nota operativa breve.

            Partido:
            - {match_name}

            Fuentes:
            - Bookmaker: home={bookmaker.home:.3f}, draw={bookmaker.draw:.3f}, away={bookmaker.away:.3f}
            - Modelo ML: home={model_probs.home:.3f}, draw={model_probs.draw:.3f}, away={model_probs.away:.3f}
            {polymarket_text}

            Quiero:
            1. Un resumen corto de la divergencia principal.
            2. La hipotesis mas razonable de por que divergen.
            3. El riesgo principal de sobreajuste o lectura equivocada.
            4. Una conclusion final en tono prudente.
            """
        ).strip()

        response = self.client.messages.create(
            model=self.model,
            max_tokens=700,
            system="Eres un analista cuantitativo de futbol. Escribes claro, prudente y sin exagerar la senal.",
            messages=[{"role": "user", "content": prompt}],
        )
        return _flatten_response_text(response)


class HeuristicAnalyst(BaseAnalyst):
    provider_name = "heuristic"

    def analyze_probability_divergence(
        self,
        match_name: str,
        bookmaker: ProbabilitySnapshot,
        model_probs: ProbabilitySnapshot,
        polymarket: ProbabilitySnapshot | None = None,
    ) -> str:
        bookmaker_map = bookmaker.as_dict()
        model_map = model_probs.as_dict()
        polymarket_map = polymarket.as_dict() if polymarket else None

        top_model, top_model_prob = model_probs.top_outcome()
        top_book, _ = bookmaker.top_outcome()
        top_poly = polymarket.top_outcome()[0] if polymarket else None

        reference = _average_reference(bookmaker_map, polymarket_map)
        diffs_vs_ref = {key: model_map[key] - reference[key] for key in model_map}
        main_outcome = max(diffs_vs_ref, key=lambda key: abs(diffs_vs_ref[key]))
        main_diff = diffs_vs_ref[main_outcome]

        confidence_gap = _top_two_gap(model_map)
        confidence_text = _describe_confidence(top_model_prob, confidence_gap)
        external_alignment = _alignment_label(top_model, top_book, top_poly)

        if main_diff > 0:
            summary = (
                f"El modelo se inclina por {OUTCOME_LABELS[top_model]} con {confidence_text}. "
                f"La mayor diferencia frente a las referencias externas esta en {OUTCOME_LABELS[main_outcome]}, "
                f"donde el modelo va {abs(main_diff):.1%} por encima del consenso entre cuotas"
            )
            if polymarket:
                summary += " y Polymarket."
            else:
                summary += "."
        else:
            summary = (
                f"El modelo se inclina por {OUTCOME_LABELS[top_model]} con {confidence_text}, "
                f"pero el mercado valora mas {OUTCOME_LABELS[main_outcome]} por un margen de {abs(main_diff):.1%}."
            )

        hypothesis = _build_hypothesis(
            top_model=top_model,
            top_book=top_book,
            top_poly=top_poly,
            main_outcome=main_outcome,
            main_diff=main_diff,
            polymarket_available=polymarket is not None,
        )
        risk = _build_risk_text(
            confidence_gap=confidence_gap,
            top_model=top_model,
            main_diff=main_diff,
            polymarket_available=polymarket is not None,
            alignment=external_alignment,
        )
        conclusion = _build_conclusion(
            top_model=top_model,
            confidence_gap=confidence_gap,
            main_diff=main_diff,
            alignment=external_alignment,
        )

        return "\n".join(
            [
                f"Reporte heuristico local para {match_name}",
                "",
                f"1. Resumen: {summary}",
                f"2. Hipotesis: {hypothesis}",
                f"3. Riesgo principal: {risk}",
                f"4. Conclusion: {conclusion}",
            ]
        )


def build_analyst(api_key: str | None, model: str) -> BaseAnalyst:
    if api_key:
        try:
            return ClaudeAnalyst(api_key=api_key, model=model)
        except Exception:
            return HeuristicAnalyst()
    return HeuristicAnalyst()


OUTCOME_LABELS = {
    "home": "victoria local",
    "draw": "empate",
    "away": "victoria visitante",
}


def _average_reference(bookmaker: dict[str, float], polymarket: dict[str, float] | None) -> dict[str, float]:
    if polymarket is None:
        return dict(bookmaker)

    return {
        key: (bookmaker[key] + polymarket[key]) / 2.0
        for key in bookmaker
    }


def _top_two_gap(probabilities: dict[str, float]) -> float:
    ordered = sorted(probabilities.values(), reverse=True)
    if len(ordered) < 2:
        return 0.0
    return ordered[0] - ordered[1]


def _describe_confidence(top_prob: float, gap: float) -> str:
    if top_prob >= 0.60 or gap >= 0.12:
        return "conviccion relativamente alta"
    if top_prob >= 0.48 or gap >= 0.06:
        return "conviccion media"
    return "conviccion baja"


def _alignment_label(top_model: str, top_book: str, top_poly: str | None) -> str:
    if top_poly is None:
        return "aligned" if top_model == top_book else "split"

    external = [top_book, top_poly]
    matches = sum(value == top_model for value in external)
    if matches == 2:
        return "strong"
    if matches == 1:
        return "mixed"
    return "opposed"


def _build_hypothesis(
    top_model: str,
    top_book: str,
    top_poly: str | None,
    main_outcome: str,
    main_diff: float,
    polymarket_available: bool,
) -> str:
    if polymarket_available and top_model == top_poly and top_model != top_book:
        return (
            "El modelo y Polymarket parecen recoger una lectura similar del partido, "
            "mientras que la cuota tradicional se mantiene algo mas conservadora. "
            "Eso suele pasar cuando el mercado de apuestas tarda mas en reflejar una narrativa reciente "
            "o cuando el modelo esta premiando forma y contexto estructural."
        )

    if polymarket_available and top_model != top_book and top_model != top_poly:
        return (
            "La divergencia sugiere que el modelo esta empujando mas de la cuenta una senal historica "
            "que no queda confirmada por las referencias externas. "
            f"La sobreponderacion parece concentrarse en {OUTCOME_LABELS[main_outcome]}."
        )

    if main_diff > 0:
        return (
            "La lectura mas probable es que el modelo este valorando mas la forma reciente, el diferencial ELO "
            "o el descanso relativo que las cuotas de mercado. "
            "Eso puede ser util si la senal es real, pero tambien puede magnificar patrones historicos que no se repiten hoy."
        )

    return (
        "Las referencias externas parecen incorporar un contexto que el modelo no captura del todo, "
        "como bajas, rotaciones, incentivos competitivos o simple incertidumbre de partido. "
        "La discrepancia aconseja no leer la salida del modelo como verdad aislada."
    )


def _build_risk_text(
    confidence_gap: float,
    top_model: str,
    main_diff: float,
    polymarket_available: bool,
    alignment: str,
) -> str:
    if confidence_gap < 0.05:
        return (
            "La distribucion esta bastante comprimida y la ventaja del primer resultado sobre el segundo es pequena. "
            "En este escenario es facil convertir una preferencia leve en una conclusion demasiado fuerte."
        )

    if abs(main_diff) >= 0.12 and alignment in {"split", "opposed"}:
        return (
            "La senal del modelo es agresiva frente al mercado y no esta bien confirmada por fuentes externas. "
            "Ese es el patron mas tipico de sobreajuste o de feature dominante mal calibrada."
        )

    if top_model == "draw":
        return (
            "El empate suele ser la clase menos estable y mas sensible a pequenas variaciones de probabilidad. "
            "Aunque aparezca arriba, conviene tratarlo como una lectura fragil."
        )

    if not polymarket_available:
        return (
            "Falta una segunda referencia externa para contrastar la narrativa del modelo. "
            "Sin Polymarket, la divergencia queda medida casi solo contra bookmaker."
        )

    return (
        "El mayor riesgo es confundir una alineacion parcial entre fuentes con una ventaja real. "
        "La coincidencia en direccion no garantiza que el tamano del edge sea explotable."
    )


def _build_conclusion(top_model: str, confidence_gap: float, main_diff: float, alignment: str) -> str:
    outcome_text = OUTCOME_LABELS[top_model]

    if alignment == "strong" and confidence_gap >= 0.06 and main_diff > 0:
        return (
            f"La lectura prudente es una inclinacion moderada hacia {outcome_text}. "
            "Hay base para justificar la direccion, pero no para tratarla como una posicion fuerte sin mas validacion."
        )

    if alignment in {"split", "mixed"}:
        return (
            f"La direccion del modelo favorece {outcome_text}, pero la dispersion entre fuentes pide cautela. "
            "Lo sensato es leerlo como lean tactico, no como conviccion alta."
        )

    if alignment == "opposed":
        return (
            f"El modelo apunta a {outcome_text}, pero el mercado no acompana. "
            "La conclusion prudente es vigilar la senal y evitar una confianza excesiva."
        )

    if main_diff <= 0:
        return (
            f"Aunque el modelo deja arriba {outcome_text}, la ventaja no parece suficientemente limpia. "
            "La postura mas sana es no sobrerreaccionar."
        )

    return (
        f"Hay una ligera preferencia por {outcome_text}, pero el edge parece fino. "
        "Mejor interpretarlo como apoyo analitico que como decision automatica."
    )


def _flatten_response_text(response: anthropic.types.Message) -> str:
    parts: list[str] = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()
