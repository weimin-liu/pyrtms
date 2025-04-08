import re

from shiny import App, ui, reactive, render
import pandas as pd
import os
import shiny
import matplotlib.pyplot as plt
import multiprocessing as mp
import h5py
import numpy as np

# Assume your reader and pipeline are in the same folder or in your PYTHONPATH
from pyrtms.rtmsBrukerMCFReader import RtmsBrukerMCFReader, Pipeline, BatchProcessor
import threading
import time
import webbrowser

app_ui = ui.page_fluid(
    ui.h2("RTMS Data Processor"),
    ui.layout_columns(
        ui.input_text("data_path", "Enter path to Bruker .d folder"),
            ui.input_numeric(
                "n_jobs",
                "Number of parallel jobs",
                value=mp.cpu_count(),
            ),
        ui.input_action_button(
            "to_hdf5",
            "Convert to HDF5",
            class_="btn-primary",
        )
    ),

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
    ui.layout_columns(
    ui.input_action_button("run_btn", "Run Pipeline", class_="btn-primary"),
        ui.input_checkbox(
            "plot_results",
            "Plot results",
            value=True
        ),
        ui.input_checkbox(
            "save_csv",
            "Save results as CSV",
            value=True
        ),
    ),
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

    @reactive.Effect
    @reactive.event(input.to_hdf5)
    def _():
        if not os.path.isdir(input.data_path()):
            print(f"Invalid file path: {input.data_path()}")
            status_text.set("Error: Invalid file path.")
            return

        try:
            reader = RtmsBrukerMCFReader.from_dir(str(input.data_path()))
            spotnumbers = reader.get_spots()['SpotNumber'].values
            spot_x = [re.findall(r"X(\d+)", str(spot))[0] for spot in spotnumbers]
            spot_x = [int(x) for x in spot_x]
            spot_y = [re.findall(r"Y(\d+)", str(spot))[0] for spot in spotnumbers]
            spot_y = [int(y) for y in spot_y]

            spot = np.column_stack((spot_x, spot_y))

            # conver every 2000 spectra to HDF5
            for i in range(0, len(reader.spotTable.index), 2000):
                part_spectra = BatchProcessor(reader,
                                             n_jobs=input.n_jobs(),
                                             show_progress=False,
                                             ).get_mul_spectra(reader.spotTable.index[i:i + 2000])
                with h5py.File(os.path.join(input.data_path(), f"data_{i // 2000}.h5"), "w") as f:
                    dset = f.create_dataset(
                        'intensities',
                        shape=(len(reader.spotTable.index), part_spectra[0].shape[0]),
                        dtype='int64',
                        chunks=(1, part_spectra[0].shape[0]),
                    )
                    for i, spectrum in enumerate(part_spectra):
                        dset[i, :] = spectrum[:,1]
                    if i//2000 == 0:
                        f.create_dataset("spot",
                                         data=spot)
                        f.create_dataset(
                            'mzs',
                            data=part_spectra[0][:, 0],
                            dtype='float64',
                        )

            status_text.set(f"Data converted to HDF5 and saved to {os.path.join(input.data_path(), 'data.h5')}")
        except Exception as e:
            status_text.set(f"Error: {e}")

    @reactive.effect
    @reactive.event(input.run_btn)
    def _():
        if not os.path.isdir(input.data_path()):
            print(f"Invalid file path: {input.data_path()}")
            status_text.set("Error: Invalid file path.")
            return

        try:
            reader = RtmsBrukerMCFReader.from_dir(str(input.data_path()))
            pipe = Pipeline(reader, n_jobs=input.n_jobs())
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
            if input.plot_results():
                calibration_fig.set(pipe.plot_calibration())
                twod_fig.set(result.viz2D())
            result_df.set(result.to_df())
            # save the DataFrame to a CSV file
            if input.save_csv():
                result.to_df().to_csv(os.path.join(input.data_path(), 'result.csv'), index=False)

            status_text.set(f"Pipeline completed successfully, results saved to {os.path.join(input.data_path(), 'result.csv')}")
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

def main():
    app = App(app_ui, server)

    def start_server():
        shiny.run_app(app, port=61235)

    threading.Thread(target=start_server, daemon=True).start()
    time.sleep(1)
    webbrowser.open("http://localhost:61235")

    # Prevent the main thread from exiting
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")


if __name__ == "__main__":
    main()


