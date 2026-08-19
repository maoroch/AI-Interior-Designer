"""
Экспорт проекта в PDF: план (упрощённая 2D-схема), список мебели по комнатам,
приблизительная смета. Часть MVP по исходному ТЗ (пункт "Возможности экспорта"),
реализована после основного пайплайна — раньше в проекте отсутствовала.

Смета — ГРУБАЯ оценка по фиксированной таблице цен за категорию мебели
(`PRICE_ESTIMATES_USD`), не привязана к реальным ценам поставщиков.
Это заведомое упрощение MVP: честная смета требует каталога с реальными
SKU и ценами, что явно исключено из MVP по ТЗ ("без каталога мебели").
"""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.models.project import Project
from app.models.scene import Scene

# Стандартные шрифты reportlab (Helvetica и т.п.) не содержат кириллических
# глифов — кириллический текст рендерился бы чёрными прямоугольниками.
# Регистрируем DejaVu Sans (свободная лицензия, поддерживает кириллицу),
# шрифт лежит в репозитории (app/export/fonts/), чтобы не зависеть от того,
# какие шрифты установлены в контейнере деплоя.
_FONTS_DIR = Path(__file__).parent / "fonts"
FONT_REGULAR = "DejaVuSans"
FONT_BOLD = "DejaVuSans-Bold"

pdfmetrics.registerFont(TTFont(FONT_REGULAR, str(_FONTS_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont(FONT_BOLD, str(_FONTS_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuSans-Oblique", str(_FONTS_DIR / "DejaVuSans-Oblique.ttf")))

# Грубые ориентировочные цены в USD по категории мебели (см. докстринг модуля).
PRICE_ESTIMATES_USD: dict[str, float] = {
    "sofa": 900,
    "bed": 700,
    "wardrobe": 550,
    "dining_table": 450,
    "desk": 280,
    "tv_stand": 220,
    "armchair": 350,
    "coffee_table": 180,
    "bookshelf": 260,
    "nightstand": 90,
    "dresser": 320,
}
DEFAULT_PRICE_ESTIMATE_USD = 150


def _estimate_price(furniture_type: str) -> float:
    return PRICE_ESTIMATES_USD.get(furniture_type, DEFAULT_PRICE_ESTIMATE_USD)


def _draw_floor_plan_schematic(scene: Scene, drawing_width_pt: float = 400) -> "Drawing":  # noqa: F821
    """Простая 2D-схема плана сверху: контуры комнат, стены, проёмы, мебель-прямоугольники."""
    from reportlab.graphics.shapes import Drawing, Rect, String

    all_x = [p[0] for room in scene.rooms for p in room.polygon] or [0, 1]
    all_y = [p[1] for room in scene.rooms for p in room.polygon] or [0, 1]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    span_x = max(max_x - min_x, 0.1)
    span_y = max(max_y - min_y, 0.1)

    scale = drawing_width_pt / span_x
    drawing_height_pt = span_y * scale
    drawing = Drawing(drawing_width_pt, drawing_height_pt)

    def to_px(x_m: float, y_m: float) -> tuple[float, float]:
        # Y инвертируем — в отчёте "верх" плана сверху страницы.
        return (x_m - min_x) * scale, drawing_height_pt - (y_m - min_y) * scale

    for room in scene.rooms:
        xs = [p[0] for p in room.polygon]
        ys = [p[1] for p in room.polygon]
        x0, y0 = to_px(min(xs), max(ys))
        w = (max(xs) - min(xs)) * scale
        h = (max(ys) - min(ys)) * scale
        drawing.add(Rect(x0, y0, w, h, fillColor=colors.HexColor("#f2efe8"), strokeColor=colors.HexColor("#333333"), strokeWidth=1.2))
        drawing.add(String(x0 + 4, y0 + h - 12, room.type.value, fontSize=7, fontName=FONT_REGULAR, fillColor=colors.HexColor("#555555")))

    for item in scene.furniture:
        x_m, _, z_m = item.position
        w_m, _, d_m = item.dimensions
        x0, y0 = to_px(x_m - w_m / 2, z_m + d_m / 2)
        drawing.add(
            Rect(
                x0,
                y0,
                w_m * scale,
                d_m * scale,
                fillColor=colors.HexColor("#c9a06a"),
                strokeColor=colors.HexColor("#7a5c36"),
                strokeWidth=0.5,
            )
        )

    return drawing


def _build_styles():
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle("TitleRu", parent=base["Title"], fontName=FONT_BOLD),
        "Normal": ParagraphStyle("NormalRu", parent=base["Normal"], fontName=FONT_REGULAR),
        "Heading2": ParagraphStyle("Heading2Ru", parent=base["Heading2"], fontName=FONT_BOLD),
        "Italic": ParagraphStyle("ItalicRu", parent=base["Italic"], fontName="DejaVuSans-Oblique"),
    }


def generate_project_pdf(project: Project, scene: Scene) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = _build_styles()
    story = []

    story.append(Paragraph("AI Interior Designer — отчёт по проекту", styles["Title"]))
    story.append(
        Paragraph(
            f"Проект: {project.id} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Вариант: {scene.variant_label} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Стиль: {scene.style or '—'} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Дата: {datetime.utcnow().strftime('%d.%m.%Y')}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 16))

    story.append(Paragraph("Схема планировки", styles["Heading2"]))
    story.append(_draw_floor_plan_schematic(scene))
    story.append(Spacer(1, 16))

    story.append(Paragraph("Список мебели", styles["Heading2"]))
    furniture_rows = [["Комната", "Предмет", "Материал", "Габариты (м)", "Оценка, $"]]
    total_cost = 0.0
    room_by_id = {r.id: r for r in scene.rooms}
    for item in scene.furniture:
        room_label = room_by_id[item.room_id].type.value if item.room_id in room_by_id else "—"
        price = _estimate_price(item.type)
        total_cost += price
        w, h, d = item.dimensions
        furniture_rows.append(
            [
                room_label,
                item.type,
                item.material or item.color or "—",
                f"{w:.2f}×{h:.2f}×{d:.2f}",
                f"{price:,.0f}",
            ]
        )

    if len(furniture_rows) == 1:
        story.append(Paragraph("Мебель ещё не сгенерирована для этого варианта.", styles["Normal"]))
    else:
        table = Table(furniture_rows, colWidths=[3 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm, 2.5 * cm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f1f1f")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, -1), FONT_REGULAR),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f5f0")]),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(table)

    story.append(Spacer(1, 16))
    story.append(Paragraph("Смета (ориентировочно)", styles["Heading2"]))
    story.append(
        Paragraph(
            "Оценка построена по усреднённым ценам за категорию мебели, "
            "без привязки к конкретным моделям или поставщикам — "
            "ориентир для планирования бюджета, не коммерческое предложение.",
            styles["Italic"],
        )
    )
    story.append(Spacer(1, 6))
    budget = project.preferences.budget_usd if project.preferences else None
    summary_rows = [["Итого по мебели, $", f"{total_cost:,.0f}"]]
    if budget:
        diff = budget - total_cost
        status = "в рамках бюджета" if diff >= 0 else "превышение бюджета"
        summary_rows.append(["Заявленный бюджет, $", f"{budget:,.0f}"])
        summary_rows.append([status, f"{abs(diff):,.0f}"])
    summary_table = Table(summary_rows, colWidths=[8 * cm, 4 * cm])
    summary_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), FONT_REGULAR),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
                ("FONTNAME", (0, 0), (0, 0), FONT_BOLD),
            ]
        )
    )
    story.append(summary_table)

    doc.build(story)
    return buffer.getvalue()
