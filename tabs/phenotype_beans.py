from shiny import ui, render, reactive
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import base64
import asyncio
from io import BytesIO
import utils
import bean_phenotyper

BEAN_COLUMNS = [
    "image",
    "bean_id",
    "width",
    "height",
    "ratio_h_w",
    "mean_r",
    "mean_g",
    "mean_b",
    "color_variance",
]


def get_ui():
    """
    Returns the UI components for the Phenotype Beans tab.

    Returns:
        ui.nav_panel: A Shiny navigation panel containing the bean phenotyping tab UI
    """
    return ui.nav_panel(
        "Phenotype Beans",
        ui.tags.script(utils.fileIterator),
        ui.input_action_button("select_dir", "Select Directory", onclick="selectDirectory()"),
        ui.br(),
        ui.output_ui("image_display"),
        ui.output_ui("completion_message"),
        ui.output_ui("processing_done"),
        ui.download_button("downloadResults", "Download .csv of bean measurements"),
    )


def register_server(
    input,
    output,
    session,
    bean_data,
    processed_image,
    processing_done_counter,
    processing_error,
    model_status,
):
    """
    Registers all server-side logic for the Phenotype Beans tab.

    Args:
        input: Shiny input object containing user inputs
        output: Shiny output object for rendering UI elements
        session: Shiny session object for managing session state
        bean_data: Reactive value storing DataFrame of bean measurements
        processed_image: Reactive value storing base64-encoded processed image
        processing_done_counter: Reactive value used to signal when processing is complete
        processing_error: Reactive value storing the latest image processing error
        model_status: Reactive value with SAM3 load status ("downloading", "ready", or "error: ...")
    """

    @reactive.effect
    async def process_current_image():
        """
        Processes the current image to segment beans and extract measurements.
        """
        if not input.current_image():
            processed_image.set(None)
            return

        if model_status.get() != "ready":
            return

        current_index = input.current_index() + 1
        total_images = input.total_images()

        with ui.Progress(min=1, max=total_images) as p:
            p.set(
                current_index,
                message=f"Processing image {current_index} of {total_images}",
                detail="Segmenting beans and measuring phenotypes...",
            )

            print("loading image")
            image_data = base64.b64decode(input.current_image())
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            try:
                overlay_rgb, records = bean_phenotyper.phenotype_image(
                    img, input.current_image_name()
                )
                overlay_rgb = cv2.resize(overlay_rgb, (0, 0), fx=0.25, fy=0.25)
                processing_error.set(None)
            except Exception as exc:
                error_message = f"{type(exc).__name__}: {exc}"
                print(f"Error processing image: {error_message}")
                processing_error.set(error_message)
                overlay_rgb = cv2.resize(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (0, 0), fx=0.25, fy=0.25
                )
                records = []

            if records:
                with reactive.isolate():
                    bean_data.set(
                        pd.concat(
                            [bean_data.get(), pd.DataFrame(records)],
                            ignore_index=True,
                        )
                    )

            fig = plt.figure(figsize=(10, 8))
            plt.imshow(overlay_rgb)
            plt.axis("off")
            plt.title(input.current_image_name())

            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode()

            processed_image.set(img_str)
            with reactive.isolate():
                processing_done_counter.set(processing_done_counter.get() + 1)

    @output
    @render.ui
    def image_display():
        """
        Renders the processed image with segmented beans highlighted.
        """
        if processed_image.get() is None:
            if model_status.get() == "downloading":
                return ui.p("Waiting for the SAM3 model to finish loading...")
            if model_status.get().startswith("error"):
                return ui.p("Fix the model loading error above before analyzing images.")
            return ui.p("Select a directory with images to begin analysis")

        error = processing_error.get()
        image = ui.img(src=f"data:image/png;base64,{processed_image.get()}")
        if error:
            return ui.div(
                ui.div(
                    ui.p("Image analysis failed:", style="font-weight: bold; margin-bottom: 0;"),
                    ui.p(error, style="color: #b71c1c; margin-top: 0;"),
                    style="margin-bottom: 10px; padding: 10px; background-color: #ffebee; border-radius: 5px;",
                ),
                image,
            )
        return image

    @output
    @render.ui
    def completion_message():
        """
        Displays a completion message when all images have been processed.
        """
        if input.show_completion():
            return ui.div(
                ui.h3("Processing Complete!", style="color: green;"),
                ui.p("All images have been processed."),
                style="margin: 20px 0; padding: 20px; background-color: #f0f0f0; border-radius: 5px;",
            )
        return None

    @output
    @render.ui
    def processing_done():
        """
        Creates a hidden div that signals when image processing is complete.
        """
        return ui.div(
            str(processing_done_counter.get()),
            id="processing_done",
            style="display:none;",
        )

    @render.download(filename="bean_analysis.csv")
    async def downloadResults():
        """
        Generates and downloads a CSV file containing bean measurements.
        """
        await asyncio.sleep(0.25)
        yield bean_data.get().to_csv(index=False)
