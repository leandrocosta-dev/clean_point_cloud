"""Paletas e folha de estilo (escuro e claro)."""

ESCURO = {
    "bg": "#0f1217", "painel": "#161b22", "cartao": "#1b212a", "campo": "#10151c", "borda": "#2a323d",
    "borda_forte": "#3a4553", "texto": "#e6edf3", "suave": "#8b97a6", "acento": "#3b9eff", "acento2": "#2b7fd6",
    "acento_txt": "#ffffff", "ok": "#3fb950", "aviso": "#d29922", "erro": "#f85149", "hover": "#222a35",
    "sel": "#1f3b5c", "console": "#0a0d12",
}
CLARO = {
    "bg": "#f3f5f8", "painel": "#ffffff", "cartao": "#ffffff", "campo": "#f7f9fb", "borda": "#d8dee6",
    "borda_forte": "#b9c3cf", "texto": "#1c2430", "suave": "#667385", "acento": "#1a73e8", "acento2": "#1560c4",
    "acento_txt": "#ffffff", "ok": "#1a7f37", "aviso": "#9a6700", "erro": "#cf222e", "hover": "#eaeef3",
    "sel": "#d6e6fb", "console": "#f7f9fb",
}


def qss(p):
    return f"""
* {{ font-family: "Segoe UI", "Inter", sans-serif; font-size: 13px; color: {p['texto']}; }}
QMainWindow, QDialog {{ background: {p['bg']}; }}
QWidget#raiz, QWidget#pagina, QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; }}
QToolTip {{ background: {p['cartao']}; color: {p['texto']}; border: 1px solid {p['borda_forte']}; padding: 6px; }}

/* barra lateral */
QFrame#lateral {{ background: {p['painel']}; border-right: 1px solid {p['borda']}; }}
QLabel#marca {{ font-size: 17px; font-weight: 700; letter-spacing: 0.3px; }}
QLabel#marcasub {{ color: {p['suave']}; font-size: 11px; }}
QPushButton#nav {{ text-align: left; padding: 10px 14px; border: none; border-radius: 8px; background: transparent;
    color: {p['suave']}; font-weight: 600; }}
QPushButton#nav:hover {{ background: {p['hover']}; color: {p['texto']}; }}
QPushButton#nav:checked {{ background: {p['sel']}; color: {p['acento']}; }}

/* cartoes de objetivo (pagina Inicio) */
QPushButton#tarefa {{ text-align: left; padding: 14px 16px; border: 1px solid {p['borda']}; border-radius: 12px;
    background: {p['cartao']}; font-weight: 500; }}
QPushButton#tarefa:hover {{ border-color: {p['acento']}; background: {p['hover']}; }}
QPushButton#tarefa:checked {{ border: 2px solid {p['acento']}; background: {p['sel']}; }}
QPushButton#tarefa:disabled {{ color: {p['suave']}; border-color: {p['borda']}; background: transparent; }}
QLabel#grupo {{ color: {p['suave']}; font-size: 10px; font-weight: 700; letter-spacing: 1.2px; padding: 12px 14px 2px 14px; }}

/* titulos */
QLabel#h1 {{ font-size: 22px; font-weight: 700; }}
QLabel#h2 {{ font-size: 15px; font-weight: 650; }}
QLabel#suave, QLabel#ajuda {{ color: {p['suave']}; }}
QLabel#ajuda {{ font-size: 12px; }}
QLabel#req {{ color: {p['erro']}; font-weight: 700; }}
QLabel#chip {{ background: {p['sel']}; color: {p['acento']}; border-radius: 9px; padding: 2px 9px; font-size: 11px; font-weight: 600; }}
QLabel#flag {{ font-family: "Cascadia Mono", Consolas, monospace; font-size: 12.5px; font-weight: 600; }}

/* cartoes */
QFrame#cartao {{ background: {p['cartao']}; border: 1px solid {p['borda']}; border-radius: 12px; }}
QFrame#linha {{ background: transparent; border: none; border-bottom: 1px solid {p['borda']}; }}
QFrame#sep {{ background: {p['borda']}; max-height: 1px; min-height: 1px; border: none; }}

/* campos */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background: {p['campo']}; border: 1px solid {p['borda']}; border-radius: 7px; padding: 6px 9px;
    selection-background-color: {p['acento']}; selection-color: {p['acento_txt']}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {{ border-color: {p['acento']}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{ color: {p['suave']}; background: transparent; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 0; border: none; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {p['cartao']}; border: 1px solid {p['borda_forte']}; selection-background-color: {p['sel']};
    selection-color: {p['texto']}; outline: none; padding: 4px; }}
QPlainTextEdit#console {{ background: {p['console']}; font-family: "Cascadia Mono", Consolas, monospace; font-size: 12px;
    border-radius: 10px; padding: 10px; }}

/* botoes */
QPushButton {{ background: {p['cartao']}; border: 1px solid {p['borda_forte']}; border-radius: 8px; padding: 7px 14px; font-weight: 600; }}
QPushButton:hover {{ background: {p['hover']}; border-color: {p['acento']}; }}
QPushButton:pressed {{ background: {p['sel']}; }}
QPushButton:disabled {{ color: {p['suave']}; border-color: {p['borda']}; background: transparent; }}
QPushButton#primario {{ background: {p['acento']}; border: 1px solid {p['acento']}; color: {p['acento_txt']}; }}
QPushButton#primario:hover {{ background: {p['acento2']}; }}
QPushButton#primario:disabled {{ background: {p['borda']}; border-color: {p['borda']}; color: {p['suave']}; }}
QPushButton#perigo {{ color: {p['erro']}; border-color: {p['erro']}; }}
QPushButton#perigo:hover {{ background: {p['erro']}; color: #fff; }}
QPushButton#mini {{ padding: 5px 9px; font-weight: 500; }}
QPushButton#fantasma {{ background: transparent; border: none; color: {p['suave']}; }}
QPushButton#fantasma:hover {{ color: {p['acento']}; background: {p['hover']}; }}

/* checkbox */
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid {p['borda_forte']}; background: {p['campo']}; }}
QCheckBox::indicator:checked {{ background: {p['acento']}; border-color: {p['acento']}; image: none; }}
QCheckBox::indicator:disabled {{ opacity: .5; }}

/* listas e tabelas */
QTreeWidget, QListWidget, QTableWidget, QTreeView {{ background: {p['painel']}; border: 1px solid {p['borda']}; border-radius: 10px;
    outline: none; alternate-background-color: {p['campo']}; gridline-color: {p['borda']}; }}
QTreeWidget::item, QListWidget::item {{ padding: 6px 6px; border-radius: 6px; }}
QTreeWidget::item:selected, QListWidget::item:selected, QTableWidget::item:selected {{ background: {p['sel']}; color: {p['texto']}; }}
QTreeWidget::item:hover, QListWidget::item:hover {{ background: {p['hover']}; }}
QHeaderView::section {{ background: {p['painel']}; color: {p['suave']}; border: none; border-bottom: 1px solid {p['borda']};
    padding: 7px 8px; font-weight: 600; }}
QTableWidget::item {{ padding: 4px 8px; }}

/* abas */
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; color: {p['suave']}; padding: 8px 16px; border-bottom: 2px solid transparent; font-weight: 600; }}
QTabBar::tab:selected {{ color: {p['acento']}; border-bottom-color: {p['acento']}; }}
QTabBar::tab:hover {{ color: {p['texto']}; }}

/* scroll */
QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {p['borda_forte']}; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {p['suave']}; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {p['borda_forte']}; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QProgressBar {{ background: {p['campo']}; border: 1px solid {p['borda']}; border-radius: 5px; height: 8px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {p['acento']}; border-radius: 4px; }}
QStatusBar {{ background: {p['painel']}; border-top: 1px solid {p['borda']}; }}
QStatusBar::item {{ border: none; }}
QSplitter::handle {{ background: transparent; }}
QMenu {{ background: {p['cartao']}; border: 1px solid {p['borda_forte']}; padding: 4px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 5px; }}
QMenu::item:selected {{ background: {p['sel']}; }}
"""
