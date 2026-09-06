"""Design tokens and QSS builder for modern dark-first GUI (WP-5).

Palette (dark-first, light fallback): background #0F1115, surface/card #161A22,
border #232936, primary #6C8CFF, success #34D399, warning #FBBF24, danger #F87171,
text #E6E9F0, muted #9AA3B2. Radius 10-14px, 8px grid, Inter/Segoe UI Variable.

Central template: ALL QSS lives here. No scattered setStyleSheet calls elsewhere.
build_qss(tokens) returns the single stylesheet applied to the app/window.
"""

from __future__ import annotations

TOKENS_DARK = {
    "bg": "#0F1115",
    "surface": "#161A22",
    "card": "#161A22",
    "card_hover": "#1B2030",
    "border": "#232936",
    "border_hover": "#2E3650",
    "primary": "#6C8CFF",
    "primary_hover": "#7D9CFF",
    "primary_pressed": "#5A72E6",
    "success": "#34D399",
    "warning": "#FBBF24",
    "danger": "#F87171",
    "text": "#E6E9F0",
    "muted": "#9AA3B2",
    "dim": "#6B7280",
    "radius": "12px",
    "radius_sm": "10px",
    "radius_lg": "14px",
    "radius_pill": "999px",
    "shadow": "rgba(0,0,0,0.40)",
    "overlay_bg": "rgba(15,17,21,0.82)",
    "focus": "#6C8CFF",
}

TOKENS_LIGHT = {
    "bg": "#F8F9FB",
    "surface": "#FFFFFF",
    "card": "#FFFFFF",
    "card_hover": "#F3F4F6",
    "border": "#E5E7EB",
    "border_hover": "#D1D5DB",
    "primary": "#6C8CFF",
    "primary_hover": "#5A7CE6",
    "primary_pressed": "#4F6AD8",
    "success": "#059669",
    "warning": "#D97706",
    "danger": "#DC2626",
    "text": "#111827",
    "muted": "#6B7280",
    "dim": "#9CA3AF",
    "radius": "12px",
    "radius_sm": "10px",
    "radius_lg": "14px",
    "radius_pill": "999px",
    "shadow": "rgba(0,0,0,0.08)",
    "overlay_bg": "rgba(248,249,251,0.88)",
    "focus": "#6C8CFF",
}


def build_qss(tokens: dict | None = None) -> str:
    """Central QSS template — single source for every widget style."""
    t = tokens or TOKENS_DARK
    return f"""
/* ===== global ===== */
QWidget {{
    font-family: 'Inter', 'Segoe UI Variable', 'Segoe UI', system-ui, sans-serif;
    font-size: 13px;
    color: {t['text']};
}}
QMainWindow, QWidget#central, QWidget#root {{
    background: {t['bg']};
}}
QScrollArea {{
    background: transparent;
    border: none;
}}
QScrollArea > QWidget > QWidget {{
    background: transparent;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {t['border']};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {t['border_hover']};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar:horizontal {{
    height: 8px;
}}
QToolTip {{
    background: {t['surface']};
    color: {t['text']};
    border: 1px solid {t['border']};
    border-radius: 8px;
    padding: 6px 10px;
}}

/* ===== rail ===== */
QFrame#rail {{
    background: {t['surface']};
    border-right: 1px solid {t['border']};
}}
QPushButton#nav {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 10px 12px;
    text-align: left;
    color: {t['muted']};
    font-size: 13px;
    font-weight: 500;
}}
QPushButton#nav:hover {{
    background: {t['card_hover']};
    border-color: {t['border']};
    color: {t['text']};
}}
QPushButton#nav:checked {{
    background: {t['primary']}18;
    border-color: {t['primary']}30;
    color: {t['primary']};
}}
QPushButton#nav:pressed {{
    background: {t['primary']}22;
}}

/* ===== cards ===== */
QFrame#card {{
    background: {t['card']};
    border: 1px solid {t['border']};
    border-radius: {t['radius']};
}}
QFrame#card:hover {{
    border-color: {t['border_hover']};
}}
QLabel#cardTitle {{
    color: {t['text']};
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}}
QLabel#hint {{
    color: {t['muted']};
    font-size: 12px;
}}
QLabel#hintError {{
    color: {t['danger']};
    font-size: 12px;
    background: rgba(248,113,113,0.10);
    border: 1px solid rgba(248,113,113,0.22);
    border-radius: 8px;
    padding: 6px 10px;
}}
QLabel#muted {{
    color: {t['muted']};
    font-size: 12px;
}}
QLabel#captionSource {{
    color: {t['muted']};
    font-size: 12px;
}}
QLabel#captionTrans {{
    color: {t['text']};
    font-size: 13px;
    font-weight: 600;
}}

/* ===== status pill ===== */
QFrame#statusPill {{
    background: {t['surface']};
    border: 1px solid {t['border']};
    border-radius: {t['radius_pill']};
    padding: 2px 6px;
}}
QLabel#statusPillLabel {{
    font-size: 12px;
    font-weight: 600;
    color: {t['text']};
}}
QLabel#elapsed {{
    color: {t['muted']};
    font-size: 12px;
    font-family: 'Cascadia Code', 'Consolas', monospace;
}}

/* ===== buttons ===== */
QPushButton {{
    background: {t['surface']};
    border: 1px solid {t['border']};
    border-radius: {t['radius_sm']};
    padding: 7px 16px;
    color: {t['text']};
    font-weight: 500;
}}
QPushButton:hover {{
    border-color: {t['primary']}55;
    background: {t['card_hover']};
}}
QPushButton:pressed {{
    background: {t['border']};
    padding-top: 8px;
    padding-bottom: 6px;
}}
QPushButton:disabled {{
    color: {t['dim']};
    background: {t['surface']};
    border-color: {t['border']};
}}
QPushButton#primary {{
    background: {t['primary']};
    color: white;
    border: none;
    font-weight: 600;
    padding: 8px 18px;
}}
QPushButton#primary:hover {{
    background: {t['primary_hover']};
}}
QPushButton#primary:pressed {{
    background: {t['primary_pressed']};
}}
QPushButton#primary:disabled {{
    background: {t['border']};
    color: {t['dim']};
}}
QPushButton#ghost {{
    background: transparent;
    border: 1px solid {t['border']};
}}
QPushButton#ghost:hover {{
    background: {t['card_hover']};
    border-color: {t['primary']}40;
}}
QPushButton#danger {{
    background: transparent;
    border: 1px solid rgba(248,113,113,0.30);
    color: {t['danger']};
}}
QPushButton#danger:hover {{
    background: rgba(248,113,113,0.10);
    border-color: {t['danger']};
}}

/* ===== inputs ===== */
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
    background: {t['surface']};
    border: 1px solid {t['border']};
    border-radius: 8px;
    padding: 6px 10px;
    selection-background-color: {t['primary']}30;
}}
QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {t['primary']}55;
}}
QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {t['primary']};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {t['muted']};
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background: {t['surface']};
    border: 1px solid {t['border']};
    border-radius: 8px;
    selection-background-color: {t['primary']}22;
    selection-color: {t['text']};
    padding: 4px;
    outline: none;
}}
QCheckBox {{
    spacing: 8px;
    color: {t['text']};
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 5px;
    border: 1px solid {t['border']};
    background: {t['surface']};
}}
QCheckBox::indicator:hover {{
    border-color: {t['primary']}60;
}}
QCheckBox::indicator:checked {{
    background: {t['primary']};
    border-color: {t['primary']};
    image: none;
}}
QCheckBox::indicator:checked:hover {{
    background: {t['primary_hover']};
}}

/* ===== progress ===== */
QProgressBar {{
    border: 1px solid {t['border']};
    border-radius: 6px;
    background: {t['surface']};
    text-align: center;
    color: {t['muted']};
    font-size: 11px;
    height: 10px;
}}
QProgressBar::chunk {{
    background: {t['primary']};
    border-radius: 5px;
}}

/* ===== toast / overlay ===== */
QFrame#toast {{
    background: {t['overlay_bg']};
    border: 1px solid {t['border']};
    border-radius: {t['radius']};
}}
QFrame#toastSuccess {{
    background: rgba(52,211,153,0.12);
    border: 1px solid rgba(52,211,153,0.28);
    border-radius: {t['radius']};
}}
QFrame#toastError {{
    background: rgba(248,113,113,0.12);
    border: 1px solid rgba(248,113,113,0.28);
    border-radius: {t['radius']};
}}
QFrame#overlayCard {{
    background: {t['overlay_bg']};
    border: 1px solid {t['border']};
    border-radius: {t['radius']};
}}
QDialog#glass {{
    background: {t['overlay_bg']};
    border: 1px solid {t['border']};
    border-radius: {t['radius_lg']};
}}

/* ===== caption card ===== */
QFrame#captionCard {{
    background: {t['surface']};
    border: 1px solid {t['border']};
    border-radius: {t['radius']};
}}
QFrame#meterBg {{
    background: {t['surface']};
    border: 1px solid {t['border']};
    border-radius: 6px;
}}

/* ===== misc ===== */
QLabel#status_ok {{
    color: {t['success']};
}}
QLabel#status_err {{
    color: {t['danger']};
    background: rgba(248,113,113,0.12);
    border-radius: 6px;
    padding: 4px 8px;
}}
QLabel#logoTitle {{
    color: {t['text']};
    font-size: 14px;
    font-weight: 700;
}}
QLabel#logoSub {{
    color: {t['muted']};
    font-size: 11px;
}}
"""


def apply_theme(app_or_widget, dark: bool = True):
    tokens = TOKENS_DARK if dark else TOKENS_LIGHT
    qss = build_qss(tokens)
    try:
        if hasattr(app_or_widget, "setStyleSheet"):
            app_or_widget.setStyleSheet(qss)
    except Exception:
        pass
    return tokens
