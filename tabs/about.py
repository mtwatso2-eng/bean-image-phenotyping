from shiny import ui

def get_ui():
    """
    Returns the UI components for the About tab.
    
    Creates a navigation panel containing information about the Bean Phenotyper
    application, including its purpose and functionality.
    
    Returns:
        ui.nav_panel: A Shiny navigation panel containing the About tab UI
    """
    return ui.nav_panel(
        "About",
        ui.h3("About Bean Phenotyper"),
        ui.p(
            "Bean Phenotyper segments individual beans in images using SAM3 "
            "and measures size and color traits for each detected bean."
        ),
        ui.h3("Using Bean Phenotyper"),
        ui.p(
            "Select a folder of bean images. The app will segment each bean, "
            "display an overlay visualization, and produce a CSV with width, "
            "height, aspect ratio, mean RGB values, and color variance for "
            "every detected bean."
        ),
        ui.h3("Model setup"),
        ui.p(
            "SAM3 weights are downloaded automatically on server launch from "
            "Hugging Face and cached locally. Request model access at "
            "huggingface.co/facebook/sam3, then set HF_TOKEN in your environment."
        ),
    )
