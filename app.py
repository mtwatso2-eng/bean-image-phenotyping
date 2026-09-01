import model_cache  # noqa: F401 - load tokens.env before model access

import asyncio

from shiny import App, ui, render, reactive
import pandas as pd
from tabs import phenotype_beans, about
from tabs.phenotype_beans import BEAN_COLUMNS
import bean_phenotyper

app_ui = ui.page_fluid(
    ui.h2("Bean Phenotyper"),
    ui.output_ui("model_status_banner"),
    ui.navset_tab(
        phenotype_beans.get_ui(),
        about.get_ui()
    )
)

def server(input, output, session):
    """
    Main server function for the Bean Phenotyper Shiny application.

    Initializes reactive values for storing application state and registers
    server logic for all tabs. Also includes a keep-alive mechanism to maintain
    the session connection.
    """
    bean_data = reactive.value(pd.DataFrame(columns=BEAN_COLUMNS))
    processed_image = reactive.value(None)
    processing_done_counter = reactive.value(0)
    processing_error = reactive.value(None)
    model_status = reactive.value("downloading")
    model_load_started = reactive.value(False)

    @reactive.effect
    async def load_model_on_startup():
        if model_load_started.get():
            return
        model_load_started.set(True)

        try:
            await asyncio.to_thread(bean_phenotyper.warm_predictor)
            model_status.set("ready")
            print("SAM3 model ready.")
        except Exception as exc:
            model_status.set(f"error: {exc}")
            print(f"SAM3 model failed to load: {exc}")

    @reactive.effect
    def keep_alive():
        """
        Keeps the Shiny session alive by invalidating every 5 seconds.

        This prevents the session from timing out during long-running image
        processing operations.
        """
        reactive.invalidate_later(5)
        print("Keeping session alive...")

    @output
    @render.ui
    def model_status_banner():
        status = model_status.get()
        if status == "ready":
            return None
        if status == "downloading":
            return ui.div(
                ui.p("Downloading and loading the SAM3 model. This happens once per server instance."),
                style="margin: 10px 0; padding: 10px; background-color: #fff8e1; border-radius: 5px;",
            )
        return ui.div(
            ui.p(f"SAM3 model failed to load: {status.removeprefix('error: ')}"),
            ui.p(
                "Check that ONNX Runtime can download rusen/sam3-browser-int8 and "
                "that the server has enough free memory."
            ),
            style="margin: 10px 0; padding: 10px; background-color: #ffebee; border-radius: 5px;",
        )

    phenotype_beans.register_server(
        input,
        output,
        session,
        bean_data,
        processed_image,
        processing_done_counter,
        processing_error,
        model_status,
    )

app = App(app_ui, server)