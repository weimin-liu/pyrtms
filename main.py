from shiny import App, ui, reactive, render
import pandas as pd
import os
import shiny
import matplotlib.pyplot as plt

# Assume your reader and pipeline are in the same folder or in your PYTHONPATH
from pyrtms.rtmsBrukerMCFReader import RtmsBrukerMCFReader, Pipeline

app_ui = ui.page_fluid(
    ui.h2("RTMS Data Processor"),
    ui.input_text("data_path", "Enter path to Bruker .d folder"),
    ui.card(
        ui.card_header("Calibration Parameters"),
            ui.layout_columns(
                ui.input_numeric("mz", "Calibration m/z", value=477.4642, step=0.0001),
                ui.input_numeric("tol1", "Tolerance (ppm or Da)", value=0.02, step=0.1),
                ui.input_numeric("min_intensity1", "Minimum Intensity", value=1e5),
                ui.input_numeric("min_snr1", "SNR", value=0, step=0.1),
                col_widths=2,
            )),
    ui.card(
        ui.card_header("Peak Picking Parameters after Calibration"),
            ui.layout_columns(
                ui.input_text("target_mzs", "Target m/z list", value="433.3805, 477.4642"),
                ui.input_numeric("tol2", "Tolerance (ppm or Da)", value=10, step=0.1),
                ui.input_numeric("min_intensity2", "Minimum Intensity", value=1e5),
                ui.input_numeric("min_snr2", "SNR", value=0, step=0.1),
                col_widths=2,
            )),
    ui.input_action_button("run_btn", "Run Pipeline", class_="btn-primary"),
    ui.output_text("status"),
    # plot the calibration results
    ui.output_plot("calibration_plot"),
    ui.output_plot('twod_plot'),
    ui.div(
        ui.output_table("result_table", striped=True, hover=True, bordered=True),
        style="height: 300px; overflow-y: auto; border: 1px solid #ccc; padding: 4px;"
    )
)

def server(input, output, session):
    result_df = reactive.Value(pd.DataFrame())
    status_text = reactive.Value("")
    calibration_fig = reactive.Value(None)
    twod_fig = reactive.Value(None)

    @reactive.effect
    @reactive.event(input.run_btn)
    def _():
        if not os.path.isdir(input.data_path()):
            print(f"Invalid file path: {input.data_path()}")
            status_text.set("Error: Invalid file path.")
            return

        try:
            reader = RtmsBrukerMCFReader.from_dir(str(input.data_path()))
            pipe = Pipeline(reader)
            pipe.set_calib_params(mz=input.mz(),
                                  tol=input.tol1(),
                                  min_intensity=input.min_intensity1(),
                                  min_snr=input.min_snr1())

            target_mzs = input.target_mzs().split(",")
            target_mzs = [float(mz.strip()) for mz in target_mzs if mz.strip()]
            pipe.set_after_params(target_mzs=target_mzs,
                                    tol=input.tol2(),
                                    min_intensity=input.min_intensity2(),
                                    min_snr=input.min_snr2())
            pipe.process()
            result = pipe.final_result

            calibration_fig.set(pipe.plot_calibration())
            twod_fig.set(result.viz2D())
            result_df.set(result.to_df())
            # save the DataFrame to a CSV file
            result.to_df().to_csv(os.path.join(input.data_path(), 'result.csv'), index=False)

            status_text.set(f"Pipeline completed successfully, results saved to {input.data_path()}/result.csv")
        except Exception as e:
            status_text.set(f"Error: {e}")
            result_df.set(pd.DataFrame())

    @output
    @render.table
    def result_table():
        return result_df()

    @output
    @render.text
    def status():
        return status_text()

    @output
    @render.plot
    def calibration_plot():
        fig = calibration_fig()
        if fig is not None:
            plt.close(fig)
            return fig
        else:
            return None

    @output
    @render.plot
    def twod_plot():
        fig = twod_fig()
        if fig is not None:
            plt.close(fig)
            return fig
        else:
            return None


if __name__ == "__main__":


    app = App(app_ui, server)

    shiny.run_app(app, port=8002)

