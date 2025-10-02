import tkinter as tk
import ttkbootstrap as ttk
from tkinter import  messagebox,filedialog, scrolledtext


import subprocess
from pathlib import Path
import json
import os
import sys

import requests
import threading

from dataclasses import dataclass, asdict
from services.genai_service import fetch_economic_answer



@dataclass
class LSTMHyperParams:
    window_size: int = 100
    hidden_size: int = 64
    num_layers: int = 2
    lr: float = 1e-3
    epochs: int = 200
    horizon: int = 10

    @classmethod
    def from_widget(
        cls,
        *,
        ws_var: tk.IntVar,
        hs_var: tk.IntVar,
        nl_var: tk.IntVar,
        lr_var: tk.DoubleVar,
        ep_var: tk.IntVar,
        ho_var: tk.IntVar
    ):
        defaults = cls()

        def _coerce(var: tk.Variable, caster, fallback):
            try:
                value=  caster(var.get())
            except Exception:
                return fallback
            return value
        

        return cls(
            window_size=ws_var.get(),
            hidden_size=hs_var.get(),
            num_layers=nl_var.get(),
            lr=lr_var.get(),
            epochs=ep_var.get(),
            horizon=ho_var.get()
        )
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    def as_dict(self) -> dict:
        return asdict(self)


class ETFAnalysisGUI:
    def __init__(self, master, script_path: Path, python_exe: str):
        self.master = master
        self.script_path = Path(script_path)
        self.python_exe = python_exe
        self.proc = None
        self.setup_window()
        self.create_widgets()
        self.load_settings()


        self.graphics_win = None
        self.graphics_compare_win, self.graphics_single_win = None, None
        self.single_ticker_var = tk.StringVar()
        self.get_compare_selection = lambda: []



        self.prediction_win = None
        self.lstm_dir_var = tk.StringVar(value=str(self.script_path.parent / "checkpoints"))
        self.get_lstm_ticker = lambda: []
        # hyperparamètres par défaut
        defaults = LSTMHyperParams()
        self.ws_var = tk.IntVar(value=defaults.window_size)
        self.hs_var = tk.IntVar(value=defaults.hidden_size)
        self.nl_var = tk.IntVar(value=defaults.num_layers)
        self.lr_var = tk.DoubleVar(value=defaults.lr)
        self.ep_var = tk.IntVar(value=defaults.epochs)
        self.ho_var = tk.IntVar(value=defaults.horizon)




        self.econ_win = None
        self.econ_question_widget = None
        self.econ_answer_widget = None
        self.econ_submit_btn = None
        self.econ_status_var = tk.StringVar(value="")
        self._econ_worker = None



        self.math_win = None
        
        self.period_var = tk.StringVar(value="max")
        self.interval_var = tk.StringVar(value="1d")
        self.light_var = tk.BooleanVar(value=False)
        self.max_var = tk.BooleanVar(value=True)
        self.log_plots_var = tk.BooleanVar(value=False)


        # Polling pour vérifier l'état du processus
        self.after_id = None
        self.start_polling()

    def setup_window(self):
        """Configuration de la fenêtre principale"""
        self.master.title("ETF Analysis Tool - Interface Graphique")
        self.master.geometry("1400x900")
        self.master.minsize(600, 500)
        
        # Style
        # style = ttk.Style()
        # style.theme_use('clam')
        
        # Configuration des couleurs
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#F18F01',
            'background': '#F5F5F5',
            'text': '#333333'
        }

    def create_widgets(self):
        """Création de tous les widgets de l'interface"""
        # Frame principal
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configuration du grid
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)

        # Titre
        title_label = ttk.Label(main_frame, text="Financial Analysis App", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        intro_text = (
    "Bonjour, voici une application qui vous permet de voir les actions et indices "
    "de vos cours boursiers préférés. De plus, dans la section Prédiction, vous "
    "pourrez retrouver nos prédictions pour certains ETF ou indicateurs économiques "
    "comme l'inflation, le GDP et bien d'autres.\n"
    "Amusez-vous bien !\n"
    "Pour toute question, veuillez accéder au bouton « aide » en bas de la page. Merci.\n\n"
    "Killian Guillaume, M1 Mathématiques Appliquées @ TSE | Intéressé par les processus "
    "stochastiques, l'apprentissage automatique et les sciences de manière générale."
)
        self.intro_label = ttk.Label(
            main_frame,
            text=intro_text,
            justify="left",
            wraplength=740  # ~ largeur utile de la fenêtre (800px - marges)
        )
        self.intro_label.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        # (optionnel) réadapte le wrap quand la taille change
        def _auto_wrap(event):
            try:
                self.intro_label.configure(wraplength=max(400, event.width - 40))
            except Exception:
                pass
        main_frame.bind("<Configure>", _auto_wrap)

        # Frame des boutons principaux
        buttons_frame = ttk.LabelFrame(main_frame, text="Actions Principales", padding="10")
        buttons_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        buttons_frame.columnconfigure(0, weight=1)
        buttons_frame.columnconfigure(1, weight=1)

        # Boutons principaux avec styles
        self.graphics_btn = ttk.Button(buttons_frame, text="Financial Charts", 
                                      command=self.open_graphics_window,
                                      style='Primary.TButton')
        self.graphics_btn.grid(row=0, column=0, padx=(0, 5), pady=5, sticky=(tk.W, tk.E))

        self.prediction_btn = ttk.Button(buttons_frame, text="Prediction for time series", 
                                        command=self.open_prediction_window,
                                        style='Secondary.TButton')
        self.prediction_btn.grid(row=0, column=1, padx=(5, 0), pady=5, sticky=(tk.W, tk.E))

        self.econ_btn = ttk.Button(
            buttons_frame, text="Economic Question",
            command=self.open_econ_window
        )
        self.econ_btn.grid(row=1, column=0, padx=(0, 5), pady=5, sticky=(tk.W, tk.E))

        self.math_btn = ttk.Button(
            buttons_frame, text="Some financial math",
            command=self.open_math_window
        )
        self.math_btn.grid(row=1, column=1, padx=(5, 0), pady=5, sticky=(tk.W, tk.E))



    def center_child(self,window, w=700, h=500):
        window.update_idletasks()
        x=(self.master.winfo_screenwidth() // 4) - (w //4)
        y=(self.master.winfo_screenheight() // 4) - (h // 4)
        window.geometry(f"{w}x{h}+{x}+{y}")


    def open_child(self, name: str, title: str):
        current = getattr(self, f"{name}_win", None)
        if current and current.winfo_exists():
            current.deiconify()
            current.lift()
            current.focus_force()
            return current

        top = tk.Toplevel(self.master)
        top.title(title)
        top.minsize(500, 400)
        top.transient(self.master)
        self.center_child(top)

        def on_close():
            try:
                top.destroy()
            finally:
                setattr(self, f"{name}_win", None)
        top.protocol("WM_DELETE_WINDOW", on_close)
        setattr(self, f"{name}_win", top)
        return top
    


    """ Graphics window and subwindows """

    def open_graphics_window(self):
        top = self.open_child("graphics", "Graphics — Analyses et graphiques")
    # Contenu minimal
        container = ttk.Frame(top, padding=16)
        container.pack(fill="both", expand=True)

        ttk.Label(container, text="🖼️ Module Graphics",
                font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 10))
        ttk.Label(container, text="Vous etes ici dans la fenetre pour afficher / comparer n'impote quel graphique d'actif financier que vous voulez").pack(anchor="w")



        analysis_frame = ttk.LabelFrame(container, text="Type d'analyse", padding=10)
        analysis_frame.pack(fill="x", pady=10) 
        analysis_frame.columnconfigure(0, weight=1)
        analysis_frame.columnconfigure(1, weight=1)

        ttk.Button(analysis_frame, text="Analyse comparative (plusieurs actifs)",
                   command=self.open_graphics_compare_window).grid(row=0, column=0, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(analysis_frame, text="Analyse simple (un seul actif)",
                     command=self.open_graphics_single_window).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)

        row4 =ttk.Frame(container)
        row4.pack(fill="x", pady=6)
        ttk.Button(row4, text="Fermer",
                command=top.destroy).pack(side="right")

        
    def open_graphics_compare_window(self):
        self.current_graphics_action = 'compare'
        top = self.open_child("graphics_compare", "Comparer plusieurs actifs")
        container = ttk.Frame(top, padding=16)
        container.pack(fill="both", expand=True)
        # 1) Ligne texte
        ttk.Label(container, text="📊 Sélectionnez plusieurs actifs à comparer (prix normalisés, rendements, etc.).",
                font=("Arial", 12)).pack(anchor="w", pady=(0, 8))

        # 2) Ligne sélection (bouton déroulant multi-sélection)
        row2 = ttk.Frame(container); row2.pack(fill="x", pady=6)
        ttk.Label(row2, text="Actifs à comparer :").pack(side="left", padx=(0, 8))

        # Barre de recherche + liste Yahoo (multi-select)
        search_ui, get_sel = self.build_ticker_search_selector_online(container, multi=True)
        self.get_compare_selection = get_sel  # mémorise le getter pour lancer l'analyse
        search_ui.pack(side="left", fill="x", expand=True)


        # 3) Ligne bouton 'Configurer' (toggle)
        row3 = ttk.Frame(container)
        row3.pack(fill="x", pady=6)
        cfg_holder = ttk.Frame(container)
        cfg_built = {"done": False}
        def toggle_cfg():
            if cfg_holder.winfo_ismapped():
                cfg_holder.pack_forget()
            else:
                if not cfg_built["done"]:
                    self.build_config_section(cfg_holder)
                    cfg_built["done"] = True
                cfg_holder.pack(fill="x", pady=(0, 6))
        ttk.Button(row3, text="Configurer", command=toggle_cfg).pack(side="left")
       

        # 4) Section Launch + bouton Fermer (ta méthode factorisée)
        self.lunch_graphics(container, mode="graphics", show_save=True, show_close=True)

        row4 =ttk.Frame(container)
        row4.pack(fill="x", pady=6)
        ttk.Button(row4, text="Fermer",
                command=top.destroy).pack(side="right")



    def open_graphics_single_window(self):
        """Sous-fenêtre: informations sur un actif (placeholder)."""

        self.current_graphics_action = 'single'
        top = self.open_child("graphics_single", "Informations sur un actif")
        container = ttk.Frame(top, padding=16)
        container.pack(fill="both", expand=True)
        # 1) Ligne texte
        ttk.Label(container, text="🖼️ Choisissez un actif pour afficher prix, volume, rendement, volatilité, etc.",
                font=("Arial", 12)).pack(anchor="w", pady=(0, 8))

        # 2) Ligne sélection (combobox simple)
        row2 = ttk.Frame(container); row2.pack(fill="x", pady=6)
        ttk.Label(row2, text="Actif :").pack(side="left", padx=(0, 8))

        # Barre de recherche + liste Yahoo (single-select)
        search_ui, get_one = self.build_ticker_search_selector_online(container, multi=False)
        self.get_single_selection = get_one  # -> [symbol] ou []
        search_ui.pack(side="left", fill="x", expand=True)


        # 3) Ligne bouton 'Configurer' (toggle)
        row3 = ttk.Frame(container); row3.pack(fill="x", pady=6)
        cfg_holder = ttk.Frame(container)
        cfg_built = {"done": False}
        def toggle_cfg():
            if cfg_holder.winfo_ismapped():
                cfg_holder.pack_forget()
            else:
                if not cfg_built["done"]:
                    self.build_config_section(cfg_holder)
                    cfg_built["done"] = True
                cfg_holder.pack(fill="x", pady=(0, 6))
        ttk.Button(row3, text="Configurer", command=toggle_cfg).pack(side="left")

        # 4) Section Launch + bouton Fermer
        self.lunch_graphics(container, mode="graphics", show_save=True, show_close=True)

        row4 =ttk.Frame(container)
        row4.pack(fill="x", pady=6)
        ttk.Button(row4, text="Fermer",
                command=top.destroy).pack(side="right")




    """ Prediction window and subwindows """

    def open_prediction_window(self):
        top = self.open_child("prediction", "Prediction — LSTM probabiliste")

        container = ttk.Frame(top, padding=16)
        container.pack(fill="both", expand=True)

        ttk.Label(container, text="🔮 Module Prediction — LSTM probabiliste (quantiles)",
                font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 10))
        ttk.Label(container, text="Entraînez un modèle ou prédisez avec le dernier modèle chargé, en choisissant un ticker.").pack(anchor="w")

        # --- Sélection du ticker (recherche Yahoo, single) ---
        box = ttk.LabelFrame(container, text="Série cible", padding=10)
        box.pack(fill="x", pady=8)
        ttk.Label(box, text="Ticker :").pack(anchor="w")
        search_ui, get_one = self.build_ticker_search_selector_online(box, multi=False)
        self.get_lstm_ticker = get_one
        search_ui.pack(fill="x", pady=(4, 0))

        # --- Hyperparamètres ---
        hp = ttk.LabelFrame(container, text="Hyperparamètres", padding=10)
        hp.pack(fill="x", pady=8)

        ttk.Label(hp, text="window_size").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(hp, textvariable=self.ws_var, width=10).grid(row=0, column=1, padx=4)

        ttk.Label(hp, text="hidden_size").grid(row=0, column=2, sticky="w", padx=4)
        ttk.Entry(hp, textvariable=self.hs_var, width=10).grid(row=0, column=3, padx=4)

        ttk.Label(hp, text="num_layers").grid(row=0, column=4, sticky="w", padx=4)
        ttk.Entry(hp, textvariable=self.nl_var, width=10).grid(row=0, column=5, padx=4)

        ttk.Label(hp, text="lr").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(hp, textvariable=self.lr_var, width=10).grid(row=1, column=1, padx=4)

        ttk.Label(hp, text="epochs").grid(row=1, column=2, sticky="w", padx=4)
        ttk.Entry(hp, textvariable=self.ep_var, width=10).grid(row=1, column=3, padx=4)

        ttk.Label(hp, text="horizon").grid(row=1, column=4, sticky="w", padx=4)
        ttk.Entry(hp, textvariable=self.ho_var, width=10).grid(row=1, column=5, padx=4)

        for c in range(6):
            hp.columnconfigure(c, weight=0)

        # --- Dossier modèle (save/load) ---
        mdl = ttk.LabelFrame(container, text="Checkpoint modèle", padding=10)
        mdl.pack(fill="x", pady=8)
        ttk.Label(mdl, text="Dossier modèle (save/load) :").grid(row=0, column=0, sticky="w")
        ttk.Entry(mdl, textvariable=self.lstm_dir_var).grid(row=0, column=1, sticky="ew", padx=6)
        def _browse_dir():
            d = filedialog.askdirectory(initialdir=self.lstm_dir_var.get() or str(self.script_path.parent))
            if d:
                self.lstm_dir_var.set(d)
        ttk.Button(mdl, text="Parcourir…", command=_browse_dir).grid(row=0, column=2)
        mdl.columnconfigure(1, weight=1)

        # --- Actions ---
        btns = ttk.Frame(container); btns.pack(fill="x", pady=12)
        ttk.Button(btns, text="🧠 Entraîner LSTM (proba)", command=self._lstm_train).pack(side="left")
        ttk.Button(btns, text="🔮 Prédire avec modèle chargé", command=self._lstm_predict).pack(side="left", padx=8)
        ttk.Button(btns, text="Fermer", command=top.destroy).pack(side="right")


    def collect_lstm_hyperparams(self) -> LSTMHyperParams:
        """Récupère les hyperparamètres depuis les widgets"""
        return LSTMHyperParams.from_widget(
            ws_var=self.ws_var,
            hs_var=self.hs_var,
            nl_var=self.nl_var,
            lr_var=self.lr_var,
            ep_var=self.ep_var,
            ho_var=self.ho_var
        )
    
    def selected_gaphics_tickers(self) -> list[str]:
        """Récupère les tickers sélectionnés dans le module graphique (compare ou single)"""
        sel = []
        compare_getter = getattr(self, "get_compare_selection", None)
        if callable(compare_getter):
            sel.extend(compare_getter() or [])
        if (not sel and hasattr(self, 'get_single_selection') 
            and callable(self.get_single_selection)):
            sel.extend(self.get_single_selection() or [])
        return [ticker for ticker in sel if ticker]
    
    @staticmethod
    def bool_env(value:bool) -> str:
        return "1" if value else "0"


    def _get_lstm_env_common(self, action: str):
        """
        Construit les variables d'environnement pour le subprocess.
        - En 'train' : pas de ticker requis, on envoie seulement HP + SAVE_DIR.
        - En 'predict' : ticker requis, on envoie HP + LOAD_DIR + LSTM_TICKER.
        """
        # Hyperparamètres depuis l'UI

        hp = self.collect_lstm_hyperparams()

        extra = {
            "LSTM_ACTION": action,     # "train" | "predict"
            "LSTM_HP": hp.to_json(),
        }

        model_dir = (self.lstm_dir_var.get() or "").strip()

        if action == "predict":
            sel = self.get_lstm_ticker() if hasattr(self, "get_lstm_ticker") else []
            ticker = (sel[0] if sel else "").strip()
            if not ticker:
                messagebox.showwarning("Ticker manquant", "Choisissez un ticker pour lancer la PRÉDICTION.")
                return None
            extra["LSTM_TICKER"] = ticker
            if model_dir:
                if not Path(model_dir).exists():
                    messagebox.showwarning(
                        "Modèle introuvable",
                        "Le dossier de checkpoints indiqué n'existe pas encore.",
                    )
                    return None
                extra["LSTM_LOAD_DIR"] = model_dir
        else:
            # TRAIN : pas de ticker requis
            if model_dir:
                Path(model_dir).mkdir(parents=True, exist_ok=True)
                    
                extra["LSTM_SAVE_DIR"] = model_dir

        return extra

    def _lstm_train(self):
        extra = self._get_lstm_env_common("train")
        if not extra:
            return
        self.run_mode("prediction", extra_env=extra)

    def _lstm_predict(self):
        extra = self._get_lstm_env_common("predict")
        if not extra:
            return
        if "LSTM_LOAD_DIR" not in extra:
            messagebox.showwarning("Modèle non spécifié",
                                "Choisissez un dossier de modèle (checkpoint) à charger.")
            return
        self.run_mode("prediction", extra_env=extra)
        
    
    """ GENAI windows """


    def open_econ_window(self):
        """Fenêtre enfant pour le module 'Economic Question'."""
        top = self.open_child("econ", "Economic Question — Outils")

        if getattr(top, "_econ_ui_ready", False):
            if self._widget_exists(self.econ_question_widget):
                try:
                    self.econ_question_widget.focus_set()
                except Exception:
                    pass
            return top

        container = ttk.Frame(top, padding=16)
        container.pack(fill="both", expand=True)

        ttk.Label(
            container,
            text="📊 Module Economic Question",
            font=("Arial", 14, "bold"),
        ).pack(anchor="w", pady=(0, 10))
        self.econ_status_var.set("")
        ttk.Label(
            container,
            text=(
                "Posez une question économique (croissance, inflation, politique monétaire, "
                "etc.). La réponse sera générée à l'aide d'un modèle OpenAI."
            ),
            wraplength=520,
            justify="left",
        ).pack(anchor="w")
        question_frame = ttk.LabelFrame(container, text="Votre question", padding=10)
        question_frame.pack(fill="both", expand=False, pady=(12, 10))

        if not self.econ_question_widget or not self.econ_question_widget.winfo_exists():
            self.econ_question_widget = scrolledtext.ScrolledText(
                question_frame,
                height=4,
                wrap="word",
            )
        self.econ_question_widget.pack(fill="both", expand=True)
        self.econ_question_widget.delete("1.0", tk.END)

        controls = ttk.Frame(container)
        controls.pack(fill="x", pady=(0, 10))

        submit_btn = ttk.Button(
            controls,
            text="Obtenir une réponse",
            command=self.submit_econ_question,
        )
        submit_btn.pack(side="left")

        ttk.Label(controls, textvariable=self.econ_status_var).pack(
            side="left", padx=10
        )

        answer_frame = ttk.LabelFrame(container, text="Réponse", padding=10)
        answer_frame.pack(fill="both", expand=True)

        if not self.econ_answer_widget or not self.econ_answer_widget.winfo_exists():
            self.econ_answer_widget = scrolledtext.ScrolledText(
                answer_frame,
                height=12,
                state="disabled",
                wrap="word",
            )
        self.econ_answer_widget.pack(fill="both", expand=True)
        self.display_econ_answer("")

        ttk.Button(container, text="Fermer", command=top.destroy).pack(anchor="e", pady=(12, 0))

    def submit_econ_question(self):
        if not self.econ_question_widget or not self.econ_question_widget.winfo_exists():
            return

        question = self.econ_question_widget.get("1.0", tk.END).strip()
        if not question:
            messagebox.showwarning(
                "Question manquante",
                "Veuillez saisir une question économique avant d'envoyer la requête.",
            )
            return

        worker = getattr(self, "_econ_worker", None)
        if worker and worker.is_alive():
            messagebox.showinfo(
                "Requête en cours",
                "Veuillez patienter que la réponse précédente soit terminée.",
            )
            return

        self.econ_status_var.set("Consultation de l'API…")
        self.display_econ_answer("Réflexion en cours…")

        def _worker():
            answer = fetch_economic_answer(question)
            self.master.after(0, lambda: self._complete_econ_request(answer))

        self._econ_worker = threading.Thread(target=_worker, daemon=True)
        self._econ_worker.start()

    def _complete_econ_request(self, answer: str):
        self.display_econ_answer(answer)
        status = "Réponse reçue." if answer else ""
        self.econ_status_var.set(status)

    def display_econ_answer(self, text: str):
        if not self.econ_answer_widget or not self.econ_answer_widget.winfo_exists():
            return

        self.econ_answer_widget.configure(state="normal")
        self.econ_answer_widget.delete("1.0", tk.END)
        self.econ_answer_widget.insert("1.0", text)
        self.econ_answer_widget.configure(state="disabled")


    """ Maths window"""

    def open_math_window(self):
        """Fenêtre enfant pour le module 'Some financial math' (placeholder)."""
        top = self.open_child("math", "Some financial math — Outils")
        container = ttk.Frame(top, padding=16)
        container.pack(fill="both", expand=True)

        ttk.Label(container, text="➗ Module Some financial math",
                font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 10))
        ttk.Label(container, text="(À venir) Petits outils et calculs financiers.").pack(anchor="w")

        ttk.Frame(container).pack(fill="x", pady=8)  # espace
        ttk.Button(container, text="Fermer", command=top.destroy).pack(anchor="e")




    def run_mode(self, mode: str, extra_env: dict | None = None):
        """Lance un mode d'analyse avec les paramètres configurés"""
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("Déjà en cours", "Un processus est déjà en cours d'exécution.")
            return

        
        self.disable_buttons()
        

        # Construction de la commande avec les paramètres
        cmd = [
            self.python_exe, "-u", str(self.script_path), 
            "--mode", mode
        ]
        
        try:
            # Variables d'environnement pour passer les paramètres
            env = os.environ.copy()
            env.update({
                'ETF_PERIOD': self.period_var.get(),
                'ETF_INTERVAL': self.interval_var.get(),
                "ETF_LIGHT": self.bool_env(self.light_var.get()),
                "ETF_MAX": self.bool_env(self.max_var.get()),
                "ETF_LOG_PLOTS": self.bool_env(self.log_plots_var.get())
            })

            if mode == "graphics":

                sel = self.selected_gaphics_tickers()

                if sel:
                    env["ETF_TICKERS"] = ",".join(sel)
                env["ETF_ACTION"] = getattr(self, "current_graphics_action", "")  # "single" | "compare"
            else:
                # Ne pas polluer l'autre mode
                env.pop("ETF_ACTION", None)
                env.pop("ETF_TICKERS", None)

            # Prediction : vous passez déjà extra_env (LSTM_ACTION, LSTM_HP, etc.), on conserve :
            if extra_env:
                env.update(extra_env)

            self.proc = subprocess.Popen(cmd, cwd=str(self.script_path.parent), env=env)
                 
                
        except Exception as e:
            
            messagebox.showerror("Erreur de lancement", str(e))
            self.enable_buttons()
            
            self.proc = None


    def start_polling(self):
        """Démarre le polling pour vérifier l'état du processus"""
        self.poll_process()

    def poll_process(self):
        """Vérifie périodiquement si le processus est terminé"""
        if self.proc is not None:
            code = self.proc.poll()
            if code is not None:
                self.on_process_done(code)
                self.proc = None
        
        # Replanifie le prochain check
        self.after_id = self.master.after(300, self.poll_process)

    def on_process_done(self, code: int):
        """Appelé quand le processus se termine"""
        
        self.enable_buttons()
        
        if code == 0:
            messagebox.showinfo("Succès", "L'analyse s'est terminée avec succès!")
        else:
            messagebox.showerror("Erreur", f"L'analyse s'est terminée avec une erreur (code={code})")

    def stop_process(self):
        """Arrête le processus en cours"""
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.enable_buttons()
            except:
                pass
        else:
            messagebox.showinfo("Aucun processus", "Aucun processus en cours à arrêter.")

    def disable_buttons(self):
        """Désactive les boutons pendant l'exécution"""
        self.graphics_btn.config(state="disabled")
        self.prediction_btn.config(state="disabled")

    def enable_buttons(self):
        """Réactive les boutons après l'exécution"""
        self.graphics_btn.config(state="normal")
        self.prediction_btn.config(state="normal")


    def lunch_graphics(self, parent, mode="graphics", show_save=True, show_close=True):

        launch_frame = ttk.LabelFrame(parent, text="Lancer une analyse graphique", padding=10)
        launch_frame.pack(fill="x", pady=10)



        # Boutons d'action du module
        btns = ttk.Frame(launch_frame)
        btns.pack(fill="x")
        
        ttk.Button(
        btns, text="Lancer (comportement actuel)",
        command=lambda m=mode: self.run_mode(m)).pack(side="left")

        if show_save:
            ttk.Button(
                btns, text="💾 Sauvegarder config",
                command=self.save_settings).pack(side="left", padx=(8, 0))

        if show_close:
            # Ferme la fenêtre Toplevel qui contient `parent` (et pas l'appli entière)
            toplevel = parent.winfo_toplevel()
            ttk.Button(btns, text="Fermer",
                command=toplevel.destroy).pack(side="right")

        return launch_frame


    def build_config_section(self, parent):
        """Construit le bloc Configuration et l'attache à `parent`."""
        config_frame = ttk.LabelFrame(parent, text="Configuration des paramètres", padding=10)
        config_frame.pack(fill="x", pady=10)

        # Ligne 1 : Période + Intervalle
        ttk.Label(config_frame, text="Période:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Combobox(
            config_frame, textvariable=self.period_var,
            values=["1y", "2y", "5y", "10y", "ytd", "max"],
            state="readonly", width=10
        ).grid(row=0, column=1, sticky=tk.W, padx=(0, 20))

        ttk.Label(config_frame, text="Intervalle:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        ttk.Combobox(
            config_frame, textvariable=self.interval_var,
            values=["1d", "1wk", "1mo"],
            state="readonly", width=10
        ).grid(row=0, column=3, sticky=tk.W, padx=(0, 0))

        # Ligne 2 : options
        ttk.Checkbutton(config_frame, text="Light mode", variable=self.light_var).grid(
            row=1, column=0, columnspan=2, sticky=tk.W, pady=(6, 0)
        )
        ttk.Checkbutton(config_frame, text="Max data", variable=self.max_var).grid(
            row=1, column=2, columnspan=2, sticky=tk.W, pady=(6, 0)
        )
        ttk.Checkbutton(config_frame, text='Log plots', variable=self.log_plots_var).grid(
            row=1, column=3, sticky=tk.W, pady=(6, 0)
            )

        for c in range(4):
            config_frame.columnconfigure(c, weight=0)

        return config_frame


    def save_settings(self):
        """Sauvegarde les paramètres dans un fichier JSON"""
        settings = {
            'period': self.period_var.get(),
            'interval': self.interval_var.get(),
            'light': self.light_var.get(),
            'max': self.max_var.get(),
            'lstm_dir': self.lstm_dir_var.get()
        }
        self.lstm_dir_var.set(settings.get('lstm_dir', str(self.script_path.parent / "checkpoints")))
        try:
            settings_file = self.script_path.parent / "etf_settings.json"
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            messagebox.showinfo("Sauvegarde", "Configuration sauvegardée avec succès!")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la sauvegarde : {str(e)}")

    def load_settings(self):
        """Charge les paramètres depuis le fichier JSON"""
        try:
            settings_file = self.script_path.parent / "etf_settings.json"
            if settings_file.exists():
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                
                self.period_var.set(settings.get('period', '5y'))
                self.interval_var.set(settings.get('interval', '1d'))
                self.light_var.set(settings.get('light', False))
                self.max_var.set(settings.get('max', True))
                
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement de la configuration : {str(e)}")

    def build_multiselect_dropdown(self, parent, options, placeholder="Sélectionner des actifs"):
        """
        Crée un 'bouton déroulant' multi-sélection (Menubutton + Menu checkbuttons).
        Retourne (widget_menubutton, get_selected_callable).
        """
        mb = ttk.Menubutton(parent, text=placeholder)
        menu = tk.Menu(mb, tearoff=False)
        mb.configure(menu=menu)

        vars_by_opt = {}

        def update_label():
            selected = [o for o, v in vars_by_opt.items() if v.get()]
            if not selected:
                mb.config(text=placeholder)
            elif len(selected) <= 3:
                mb.config(text=", ".join(selected))
            else:
                mb.config(text=f"{len(selected)} sélectionnés")

        for opt in options:
            var = tk.BooleanVar(value=False)
            vars_by_opt[opt] = var
            menu.add_checkbutton(
                label=opt,
                variable=var,
                onvalue=True, offvalue=False,
                command=update_label
            )

        def get_selected():
            return [o for o, v in vars_by_opt.items() if v.get()]

        return mb, get_selected

    def build_ticker_search_selector_online(self, parent, multi=True,
                                 placeholder="Rechercher un ETF ou ticker…",
                                 max_results=20, debounce_ms=250):
        """
        Construit :
        - une zone 'sélectionnés' (pills amovibles)
        - une barre de recherche (Entry)
        - une Listbox de suggestions filtrées (préfixe)
        Retourne: (wrapper_frame, get_selected_callable)
        """
        # -- état interne
        selected = {}   # {ticker: pill_frame}
        display_items = []    # liste courante des résultats [{symbol,name,exch}, ...]
        last_qid = {"id": 0}  # pour éviter les MAJ hors-ordre (threading)
        scheduled = {"job": None}

        # -- conteneur principal
        wrapper = ttk.Frame(parent)
        wrapper.pack(fill="x", pady=6)

        # 0) zone des éléments sélectionnés (au-dessus de la barre de recherche)

        if multi:
            selected_holder = ttk.Frame(wrapper)
            selected_holder.pack(fill="x", pady=(0, 6))
        else:
            selected_holder = None

        def _add_pill(ticker):
            if multi:
                if ticker in selected:
                    return
                
                pill = ttk.Frame(selected_holder, padding=(6, 2))
                ttk.Label(pill, text=ticker).pack(side="left")
                ttk.Button(pill, text="×", width=2,
                        command=lambda t=ticker, f=pill: (f.destroy(), selected.pop(t, None))
                        ).pack(side="left", padx=(6, 0))
                pill.pack(side="left", padx=4, pady=2)
                selected[ticker] = pill
            else:
                selected.clear()
                selected[ticker] = None
                search_var.set(ticker)
                _update_suggestions([])

            

        # 1) ligne de recherche
        row = ttk.Frame(wrapper)
        row.pack(fill="x")
        search_var = tk.StringVar()
        entry = ttk.Entry(row, textvariable=search_var, width=32)
        entry.pack(side="left")

        def clear():
            search_var.set("")
            selected.clear()
            _update_suggestions([])

        ttk.Button(row, text="Effacer", command=clear).pack(side="left", padx=6)


        # 2) suggestions
        listbox = tk.Listbox(wrapper, height=6, exportselection=False)
        listbox.pack(fill="x", pady=(6, 0))
        listbox.forget()  # masquée tant qu'il n'y a pas de résultat

        # # -- helpers
        # lowered = [opt for opt in options]  # conserve l’ordre d’entrée

        def _filter(prefix):
            p = prefix.strip().lower()
            if not p:
                return []
            # correspondance par préfixe (début du ticker/nom)
            return [o for o in lowered if o.lower().startswith(p)][:max_results]

        def _update_suggestions(items):
            nonlocal display_items
            display_items = items[:max_results]
            listbox.delete(0, tk.END)
            if not display_items:
                if listbox.winfo_ismapped():
                    listbox.forget()
                return
            # remplit "SYMBOL — NAME (EXCH)"
            for it in display_items:
                name = f" — {it['name']}" if it.get("name") else ""
                exch = f" ({it['exch']})" if it.get("exch") else ""
                listbox.insert(tk.END, f"{it['symbol']}{name}{exch}")
            if not listbox.winfo_ismapped():
                listbox.pack(fill="x", pady=(6, 0))

        def _choose_from_list(event=None):
            if listbox.size() == 0:
                return
            idx = listbox.curselection()
            if not idx:
                # ENTER depuis l'Entry -> prendre le 1er si dispo
                if event and event.widget is entry and listbox.size() > 0:
                    it = display_items[0]
                else:
                    return
            else:
                it = display_items[idx[0]]
            _add_pill(it["symbol"])
            entry.focus_set()

        # recherche (debounced) + thread réseau
        def _debounced_search(*_):
            # annule job précédent si existant
            if scheduled["job"] is not None:
                parent.after_cancel(scheduled["job"])
            if not multi:
                selected.clear()
            scheduled["job"] = parent.after(debounce_ms, _start_search)

        def _start_search():
            prefix = search_var.get().strip()
            if not prefix:
                _update_suggestions([])
                return
            qid = last_qid["id"] = last_qid["id"] + 1

            def worker():
                items = self.yahoo_suggest(prefix, count=max_results)
                # MAJ UI côté thread principal
                try:
                    parent.after(0, lambda: (qid == last_qid["id"]) and _update_suggestions(items))
                except Exception:
                    pass

            threading.Thread(target=worker, daemon=True).start()

        # bindings
        entry.bind("<KeyRelease>", _debounced_search)
        entry.bind("<Down>", lambda e: (listbox.focus_set(),
                                        listbox.selection_clear(0, tk.END),
                                        listbox.selection_set(0), "break"))
        entry.bind("<Return>", _choose_from_list)
        listbox.bind("<Double-Button-1>", _choose_from_list)
        listbox.bind("<Return>", _choose_from_list)
        listbox.bind("<Escape>", lambda e: (listbox.forget(), entry.focus_set(), "break"))

        def get_selected():
            if multi:
                return list(selected.keys())
            val = (search_var.get() or "").strip()
            return [val] if val else []

        # focus initial
        entry.focus_set()
        return wrapper, get_selected


    def yahoo_suggest(self, prefix: str, count: int = 20):
        """Retourne une liste de dicts {symbol, name, exch} depuis l'API recherche Yahoo."""
        if not prefix:
            return []
        url = "https://query2.finance.yahoo.com/v1/finance/search"  # parfois query1
        headers = {
            "User-Agent": "Mozilla/5.0"  # certains clusters refusent sans UA
        }
        try:
            r = requests.get(url, params={"q": prefix, "quotesCount": count, "newsCount": 0}, headers=headers, timeout=6)
            r.raise_for_status()
            data = r.json()
        except Exception:
            return []

        out, seen = [], set()
        for q in data.get("quotes", []):
            sym = q.get("symbol")
            if not sym or sym in seen:
                continue
            seen.add(sym)
            name = q.get("shortname") or q.get("longname") or ""
            exch = q.get("exchDisp") or q.get("exchange") or ""
            out.append({"symbol": sym, "name": name, "exch": exch})
        return out


    def on_closing(self):
        """Appelé à la fermeture de la fenêtre"""
        if self.proc and self.proc.poll() is None:
            if messagebox.askokcancel("Fermeture", "Un processus est en cours. Voulez-vous l'arrêter et quitter ?"):
                self.stop_process()
                if self.after_id:
                    self.master.after_cancel(self.after_id)
                self.master.destroy()
        else:
            if self.after_id:
                self.master.after_cancel(self.after_id)
            self.master.destroy()


    def get_available_tickers(self):
        try:
            from etf_collector import EuropeanETFCollector
            return EuropeanETFCollector.get_tickers()
        except Exception:
            return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "^GSPC"]


def main():
    """Point d'entrée principal pour l'interface"""
    root = ttk.Window(themename="superhero")
    
    # Configuration du style
    # style = ttk.Style()
    # style.configure('Primary.TButton', background='#2E86AB', foreground='white')
    # style.configure('Secondary.TButton', background='#A23B72', foreground='white')
    
    # Création de l'interface
    app = ETFAnalysisGUI(
        root,
        script_path=Path(__file__).parent / "main.py",  # Chemin vers main.py
        python_exe=sys.executable
    )
    
    # Gestion de la fermeture
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Centrage de la fenêtre
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()