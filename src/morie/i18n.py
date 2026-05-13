"""morie.i18n — minimal English / French string surface.

Reaches only the high-traffic CLI strings (cheatsheet labels, tutorial
headings, doctor headings, friendly-error hints).  Methodology
documentation stays English-only — translating dense statistical prose
is its own scoped project.

Activation: set ``MORIE_LOCALE=fr`` before running ``morie``.  Defaults
to English when unset or set to anything other than ``fr``.

This is a deliberately lightweight i18n layer (no gettext, no .po
files) because:

  * Canadian sociolegal users are the target audience; bilingual EN/FR
    is the regulatory norm; nothing else needs to ship today.
  * gettext requires C-extension compilation on first run; we want
    pure-Python so the wheel installs anywhere.

Adding a new locale is a one-file edit: add another dict and route
through ``_select_locale``.
"""

from __future__ import annotations

import os

EN: dict[str, str] = {
    "doctor.heading": "MORIE Doctor — environment diagnostics",
    "doctor.status_ok": "OK",
    "doctor.status_warn": "WARN",
    "doctor.status_fail": "FAIL",
    "tutorial.welcome": "Welcome to the morie tutorial.",
    "tutorial.continue_prompt": "[Enter] continue · [s] skip · [q] quit: ",
    "cheatsheet.install": "INSTALL",
    "cheatsheet.learn": "LEARN",
    "cheatsheet.run": "RUN AN ANALYSIS",
    "cheatsheet.pull": "PULL DATA (no Python needed)",
    "cheatsheet.ingest": "ADVANCED INGESTION",
    "cheatsheet.help": "ASK FOR HELP",
    "cheatsheet.refs": "REFERENCE",
    "pull.wrote_n_rows": "wrote {path}  ({rows:,} rows, {cols} cols)",
    "error.network_failed": "Network call failed: {msg}",
}

FR: dict[str, str] = {
    "doctor.heading": "MORIE Doctor — diagnostics de l'environnement",
    "doctor.status_ok": "OK",
    "doctor.status_warn": "AVERT",
    "doctor.status_fail": "ÉCHEC",
    "tutorial.welcome": "Bienvenue dans le didacticiel morie.",
    "tutorial.continue_prompt": "[Entrée] continuer · [s] passer · [q] quitter : ",
    "cheatsheet.install": "INSTALLATION",
    "cheatsheet.learn": "APPRENTISSAGE",
    "cheatsheet.run": "EXÉCUTER UNE ANALYSE",
    "cheatsheet.pull": "TÉLÉCHARGER DES DONNÉES (sans Python)",
    "cheatsheet.ingest": "INGESTION AVANCÉE",
    "cheatsheet.help": "DEMANDER DE L'AIDE",
    "cheatsheet.refs": "RÉFÉRENCE",
    "pull.wrote_n_rows": "écrit {path}  ({rows:,} lignes, {cols} colonnes)",
    "error.network_failed": "L'appel réseau a échoué : {msg}",
}

ES: dict[str, str] = {
    "doctor.heading": "Doctor de morie — diagnóstico del entorno",
    "doctor.status_ok": "OK",
    "doctor.status_warn": "AVISO",
    "doctor.status_fail": "ERROR",
    "tutorial.welcome": "Bienvenido al tutorial de morie.",
    "tutorial.continue_prompt": "[Intro] continuar · [s] omitir · [q] salir: ",
    "cheatsheet.install": "INSTALAR",
    "cheatsheet.learn": "APRENDER",
    "cheatsheet.run": "EJECUTAR UN ANÁLISIS",
    "cheatsheet.pull": "OBTENER DATOS (sin necesidad de Python)",
    "cheatsheet.ingest": "INGESTA AVANZADA",
    "cheatsheet.help": "PEDIR AYUDA",
    "cheatsheet.refs": "REFERENCIA",
    "pull.wrote_n_rows": "se escribió {path}  ({rows:,} filas, {cols} columnas)",
    "error.network_failed": "Falló la llamada de red: {msg}",
}

DE: dict[str, str] = {
    "doctor.heading": "morie Doctor — Umgebungsdiagnose",
    "doctor.status_ok": "OK",
    "doctor.status_warn": "WARN",
    "doctor.status_fail": "FEHL",
    "tutorial.welcome": "Willkommen zum morie-Tutorial.",
    "tutorial.continue_prompt": "[Eingabe] weiter · [s] überspringen · [q] beenden: ",
    "cheatsheet.install": "INSTALLATION",
    "cheatsheet.learn": "LERNEN",
    "cheatsheet.run": "ANALYSE AUSFÜHREN",
    "cheatsheet.pull": "DATEN ABRUFEN (ohne Python)",
    "cheatsheet.ingest": "ERWEITERTE DATENERFASSUNG",
    "cheatsheet.help": "HILFE ANFORDERN",
    "cheatsheet.refs": "REFERENZ",
    "pull.wrote_n_rows": "{path} geschrieben  ({rows:,} Zeilen, {cols} Spalten)",
    "error.network_failed": "Netzwerkaufruf fehlgeschlagen: {msg}",
}

ZH: dict[str, str] = {
    "doctor.heading": "morie 诊断器 — 环境诊断",
    "doctor.status_ok": "正常",
    "doctor.status_warn": "警告",
    "doctor.status_fail": "失败",
    "tutorial.welcome": "欢迎使用 morie 教程。",
    "tutorial.continue_prompt": "[回车] 继续 · [s] 跳过 · [q] 退出: ",
    "cheatsheet.install": "安装",
    "cheatsheet.learn": "学习",
    "cheatsheet.run": "运行分析",
    "cheatsheet.pull": "拉取数据(无需 Python)",
    "cheatsheet.ingest": "高级数据导入",
    "cheatsheet.help": "寻求帮助",
    "cheatsheet.refs": "参考资料",
    "pull.wrote_n_rows": "已写入 {path}  ({rows:,} 行, {cols} 列)",
    "error.network_failed": "网络调用失败: {msg}",
}

PT: dict[str, str] = {
    "doctor.heading": "Doctor do morie — diagnóstico do ambiente",
    "doctor.status_ok": "OK",
    "doctor.status_warn": "AVISO",
    "doctor.status_fail": "FALHA",
    "tutorial.welcome": "Bem-vindo ao tutorial do morie.",
    "tutorial.continue_prompt": "[Enter] continuar · [s] pular · [q] sair: ",
    "cheatsheet.install": "INSTALAR",
    "cheatsheet.learn": "APRENDER",
    "cheatsheet.run": "EXECUTAR UMA ANÁLISE",
    "cheatsheet.pull": "OBTER DADOS (sem precisar de Python)",
    "cheatsheet.ingest": "INGESTÃO AVANÇADA",
    "cheatsheet.help": "PEDIR AJUDA",
    "cheatsheet.refs": "REFERÊNCIA",
    "pull.wrote_n_rows": "{path} gravado  ({rows:,} linhas, {cols} colunas)",
    "error.network_failed": "Falha na chamada de rede: {msg}",
}

JA: dict[str, str] = {
    "doctor.heading": "morie Doctor — 環境診断",
    "doctor.status_ok": "OK",
    "doctor.status_warn": "警告",
    "doctor.status_fail": "失敗",
    "tutorial.welcome": "morie チュートリアルへようこそ。",
    "tutorial.continue_prompt": "[Enter] 続行 · [s] スキップ · [q] 終了: ",
    "cheatsheet.install": "インストール",
    "cheatsheet.learn": "学習",
    "cheatsheet.run": "解析を実行",
    "cheatsheet.pull": "データ取得(Python 不要)",
    "cheatsheet.ingest": "高度なデータ取り込み",
    "cheatsheet.help": "ヘルプを求める",
    "cheatsheet.refs": "リファレンス",
    "pull.wrote_n_rows": "{path} に書き込みました  ({rows:,} 行, {cols} 列)",
    "error.network_failed": "ネットワーク呼び出しに失敗しました: {msg}",
}

AR: dict[str, str] = {
    "doctor.heading": "مُشخِّص morie — تشخيص البيئة",
    "doctor.status_ok": "سليم",
    "doctor.status_warn": "تحذير",
    "doctor.status_fail": "فشل",
    "tutorial.welcome": "مرحبًا بك في دليل morie التعليمي.",
    "tutorial.continue_prompt": "[Enter] متابعة · [s] تخطٍّ · [q] إنهاء: ",
    "cheatsheet.install": "التثبيت",
    "cheatsheet.learn": "التعلّم",
    "cheatsheet.run": "تشغيل تحليل",
    "cheatsheet.pull": "جلب البيانات (دون الحاجة إلى Python)",
    "cheatsheet.ingest": "الاستيعاب المتقدّم",
    "cheatsheet.help": "طلب المساعدة",
    "cheatsheet.refs": "المرجع",
    "pull.wrote_n_rows": "تمت كتابة {path}  ({rows:,} صفًّا، {cols} عمودًا)",
    "error.network_failed": "فشل الاتصال بالشبكة: {msg}",
}

HI: dict[str, str] = {
    "doctor.heading": "morie डॉक्टर — परिवेश निदान",
    "doctor.status_ok": "ठीक",
    "doctor.status_warn": "चेतावनी",
    "doctor.status_fail": "विफल",
    "tutorial.welcome": "morie ट्यूटोरियल में आपका स्वागत है।",
    "tutorial.continue_prompt": "[Enter] जारी रखें · [s] छोड़ें · [q] बाहर निकलें: ",
    "cheatsheet.install": "संस्थापन",
    "cheatsheet.learn": "सीखें",
    "cheatsheet.run": "विश्लेषण चलाएँ",
    "cheatsheet.pull": "डेटा प्राप्त करें (Python की आवश्यकता नहीं)",
    "cheatsheet.ingest": "उन्नत डेटा अंतर्ग्रहण",
    "cheatsheet.help": "सहायता माँगें",
    "cheatsheet.refs": "संदर्भ",
    "pull.wrote_n_rows": "{path} लिखा गया  ({rows:,} पंक्तियाँ, {cols} स्तंभ)",
    "error.network_failed": "नेटवर्क कॉल विफल: {msg}",
}

LOCALES: dict[str, dict[str, str]] = {
    "en": EN, "fr": FR, "es": ES, "de": DE, "zh": ZH,
    "pt": PT, "ja": JA, "ar": AR, "hi": HI,
}


def _select_locale() -> dict[str, str]:
    """Resolve the active locale from MORIE_LOCALE, falling back to EN.

    Honours 2-char codes ("fr") and BCP-47 prefixes ("fr-CA", "zh-Hans").
    """
    raw = os.environ.get("MORIE_LOCALE", "").lower()
    code = raw[:2]
    return LOCALES.get(code, EN)


def t(key: str, **fmt: object) -> str:
    """Translate a key; if the key isn't registered, return the key itself."""
    table = _select_locale()
    s = table.get(key, EN.get(key, key))
    if fmt:
        try:
            return s.format(**fmt)
        except (KeyError, IndexError):
            return s
    return s


def current_locale() -> str:
    """Return the two-letter locale code currently in effect ('en' or 'fr')."""
    return "fr" if _select_locale() is FR else "en"
