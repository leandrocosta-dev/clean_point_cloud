"""Linha de comando unica para o Revo Scan 6 (automacao por UI; os comandos marcados [mouse] usam o mouse real).

    python pipeline/revo_cli.py status                          # Revo aberto? lingua, grupos, scans visiveis (sem mouse)
    python pipeline/revo_cli.py captura saida.png               # imagem da janela do Revo (sem mouse)
    python pipeline/revo_cli.py biblioteca-scans [--nome X]     # scans e merges da biblioteca (arquivos; sem Revo)
    python pipeline/revo_cli.py importar PROJ.revo --grupo NOME # [mouse] + New Group, entra nele, Load Project
    python pipeline/revo_cli.py grupo NOME                      # [mouse] entra num grupo existente
    python pipeline/revo_cli.py abrir-scan NOME [--grupo G]     # [mouse] abre um scan (cartao) da biblioteca
    python pipeline/revo_cli.py fundir NOME --grupo G --ponto 0.1 [--malha alta --grade 0.1] [--exportar PASTA]
    python pipeline/revo_cli.py fechar                          # fecha o Revo (WM_CLOSE)

Fluxo completo (importar + fundir + malha + exportar todos os scans): pipeline/revo_pipeline.py --grupo NOME.
Avaliar um Merge Alignment feito no Revo (sem abrir o Revo): pipeline/avaliar_merge_revo.py.
Regras: cada importacao em GRUPO NOVO; sessao do Revo em ingles (a automacao o inicia com LANG=en_US);
nao mexer no mouse durante comandos [mouse].
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import revo_ui as R


def _ui_existente():
    import psutil
    for p in psutil.process_iter(["name"]):
        if p.info["name"] == "RevoScan6.exe":
            return R.RevoUI(p.pid)
    return None


def cmd_status(a):
    ui = _ui_existente()
    if ui is None:
        print("Revo Scan fechado")
        return
    print("pid", ui.pid, "| travado:", ui._travado(ui.pid))
    w = ui.principal
    print("janela:", w.window_text())
    textos = {d.window_text() for d in w.descendants(control_type="Text")}
    botoes = {d.window_text() for d in w.descendants(control_type="Button")}
    print("lingua:", "ingles" if ("Load Project" in textos or "File" in botoes or "Fusion" in botoes) else "outra (sessao do usuario?)")
    if "Fusion" in botoes:
        print("tela: scan aberto (Fusion/Meshing visiveis)")
    else:
        print("tela: biblioteca | scans visiveis:", ui._contagem())
        print("grupos:", ui.grupos())


def cmd_captura(a):
    ui = _ui_existente()
    if ui is None:
        sys.exit("Revo Scan fechado")
    print(ui.captura(os.path.abspath(a.saida)))


def cmd_biblioteca(a):
    linhas = []
    for d in os.listdir(R.BIBLIOTECA):
        pasta = os.path.join(R.BIBLIOTECA, d)
        for f in glob.glob(os.path.join(pasta, "*.revox")):
            nome = os.path.basename(f)[:-6]
            if a.nome and a.nome.lower() not in nome.lower():
                continue
            guid = ""
            prop = os.path.join(pasta, "property.rvproj")
            if os.path.exists(prop):
                guid = json.load(open(prop, encoding="utf-8")).get("guid", "")
            pcs = [os.path.basename(x) for x in glob.glob(os.path.join(pasta, "pointclouds", "*.ply"))]
            linhas.append((os.path.getmtime(f), d, nome, guid, pcs))
    linhas.sort(reverse=True)
    for t, d, nome, guid, pcs in linhas[:a.limite]:
        print(time.strftime("%Y-%m-%d %H:%M", time.localtime(t)), d, nome, guid[:8], pcs)


def cmd_importar(a):
    ui = R.RevoUI.abrir()
    ui.novo_grupo(a.grupo)
    achados = ui.carregar_projeto(os.path.abspath(a.revo))
    if ui._contagem() < len(achados):          # a grade do grupo nao se atualiza sozinha apos o Load Project
        ui.entrar_grupo("Ungrouped"); ui.entrar_grupo(a.grupo)
    print("grupo:", a.grupo)
    for nome, pasta in sorted(achados.items()):
        print(" ", nome, pasta)
    print("scans no grupo:", ui._contagem())
    R.restaurar_janelas()


def cmd_grupo(a):
    ui = R.RevoUI.abrir()
    ui.entrar_grupo(a.nome)
    print("scans no grupo:", ui._contagem())
    R.restaurar_janelas()


def cmd_abrir_scan(a):
    ui = R.RevoUI.abrir()
    print("aberto:", ui.abrir_scan(a.nome, grupo=a.grupo))
    R.restaurar_janelas()


def cmd_fundir(a):
    # parametros sao lidos so na 1a abertura do scan na sessao: grava ANTES de abrir (revox achado pelo group.cfg)
    revox = R.revox_no_grupo(a.grupo, a.nome) if a.grupo else R.pasta_mais_recente(a.nome) + os.sep + a.nome + ".revox"
    R.parametros(a.nome, ponto_mm=a.ponto, malha=a.malha, grade_mm=a.grade, revox=revox)
    ui = R.RevoUI.abrir()
    ui.abrir_scan(a.nome, os.path.dirname(revox), grupo=a.grupo)
    r = {"fusao": ui.fundir()}
    if a.malha:
        r["malha"] = ui.malha()
    if a.exportar:
        r["arquivos"] = ui.exportar(a.exportar, prefixos=("Fused", "Mesh") if a.malha else ("Fused",))
    print(json.dumps(r, ensure_ascii=False))
    R.restaurar_janelas()


def _listar_controles(ui):
    for w in ui.ds.windows(process=ui.pid):
        print("JANELA", repr(w.window_text()), w.class_name(), w.rectangle())
        for d in w.descendants():
            t = d.window_text(); ct = d.element_info.control_type
            if d.is_visible() and ct in ("Button", "Text", "CheckBox", "ListItem", "RadioButton", "TabItem", "Edit", "ComboBox") \
                    and (t or ct in ("CheckBox", "Edit")):
                print("  ", repr(t), ct, d.element_info.class_name, d.rectangle())


def cmd_tela(a):
    ui = _ui_existente()
    if ui is None:
        sys.exit("Revo Scan fechado")
    _listar_controles(ui)
    if a.captura:
        print("captura:", ui.captura(os.path.abspath(a.captura)))


def cmd_clicar(a):
    ui = R.RevoUI.abrir()
    if a.grupo:
        ui.entrar_grupo(a.grupo)
    alvo = next((d for d in ui.principal.descendants() if d.is_visible() and d.window_text() == a.texto), None)
    if alvo is None:
        sys.exit(f"controle {a.texto!r} nao encontrado")
    alvo.click_input()
    time.sleep(a.espera)
    _listar_controles(ui)
    if a.captura:
        print("captura:", ui.captura(os.path.abspath(a.captura)))
    R.restaurar_janelas()


def cmd_fechar(a):
    ui = _ui_existente()
    if ui:
        ui.fechar()
    print("fechado")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="estado do Revo (sem mouse)").set_defaults(f=cmd_status)
    p = sub.add_parser("captura", help="imagem da janela (sem mouse)"); p.add_argument("saida"); p.set_defaults(f=cmd_captura)
    p = sub.add_parser("biblioteca-scans", help="scans/merges da biblioteca por data (sem Revo)")
    p.add_argument("--nome"); p.add_argument("--limite", type=int, default=30); p.set_defaults(f=cmd_biblioteca)
    p = sub.add_parser("importar", help="[mouse] grupo novo + Load Project")
    p.add_argument("revo"); p.add_argument("--grupo", required=True); p.set_defaults(f=cmd_importar)
    p = sub.add_parser("grupo", help="[mouse] entra num grupo"); p.add_argument("nome"); p.set_defaults(f=cmd_grupo)
    p = sub.add_parser("abrir-scan", help="[mouse] abre um scan")
    p.add_argument("nome", help="nome do cartao/revox, ex. 20261108_094322_COPIA"); p.add_argument("--grupo"); p.set_defaults(f=cmd_abrir_scan)
    p = sub.add_parser("fundir", help="[mouse] abre, grava parametros, funde (e malha/exporta)")
    p.add_argument("nome"); p.add_argument("--grupo"); p.add_argument("--ponto", type=float)
    p.add_argument("--malha", choices=list(R.MALHA_MODO)); p.add_argument("--grade", type=float)
    p.add_argument("--exportar", help="pasta de saida dos .ply"); p.set_defaults(f=cmd_fundir)
    p = sub.add_parser("tela", help="lista controles visiveis (sem mouse)"); p.add_argument("--captura"); p.set_defaults(f=cmd_tela)
    p = sub.add_parser("clicar", help="[mouse] clica num controle pelo texto e lista a tela seguinte")
    p.add_argument("texto"); p.add_argument("--grupo"); p.add_argument("--espera", type=float, default=3.0)
    p.add_argument("--captura"); p.set_defaults(f=cmd_clicar)
    sub.add_parser("fechar", help="fecha o Revo").set_defaults(f=cmd_fechar)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
