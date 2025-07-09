import os
import threading
import time
import webbrowser

import numpy as np
from shiny import App, ui, reactive, run_app
from shinywidgets import output_widget, render_widget
import plotly.express as px
import pandas as pd


from pyrtms.mz_finder import Config, profile_to_line, get_kde_curve, get_target_mz, get_eic

app_ui = ui.page_fluid(
    ui.card(
        ui.card_header('Load .d folder'),
        ui.input_text("data_path", label=None, value='path/to/.d/folder/', width='80%'),
        ui.input_action_button('load_d', label='Load', width='10%'),
    ),

    ui.card(
        ui.card_header('Get measured m/z'),
    ui.row(
ui.column(5, ui.input_numeric('target_mz',label='Target m/z', value=None),
    ui.row(
    ui.column(2, ui.input_action_button('get_measured_mz', label='Go')),
    ui.column(3,
              ui.input_checkbox('plot_kde_mz', label='Plot result',value=True),
              ui.input_checkbox('create_eic', label='Create EIC',value=False),
              ui.input_checkbox('plot_eic', label='Plot EIC',value=False)
              )
    )
          ),
    ui.column(7,
              output_widget('kde_mz_plot'),
              output_widget('eic_plot')))

),

)

def server(input, output, session):
    mass_spec_config = Config()
    x_val = reactive.Value(None)
    y_val = reactive.Value(None)
    kde_fig_to_display = reactive.Value(None)
    eic = reactive.Value(None)
    eic_fig_to_display = reactive.Value(None)

    @reactive.effect
    @reactive.event(input.load_d)
    def calculate_kde_on_mz():
        if not os.path.exists(os.path.join(input.data_path(), mass_spec_config.line_spectra_filename)):
            profile_to_line(input.data_path(), mass_spec_config)
        x, y = get_kde_curve(input.data_path(), mass_spec_config)
        x_val.set(x)
        y_val.set(y)

    @reactive.effect
    @reactive.event(input.get_measured_mz)
    def calculate_measured_mz_on_target_mz():
        measured_mz, measured_da_tol, fig = get_target_mz(input.target_mz(), x_val.get(), y_val.get(), mass_spec_config, plot=True)
        if input.plot_kde_mz():
            kde_fig_to_display.set(fig)
        if input.create_eic():
            eic.set(get_eic(measured_mz, measured_da_tol, input.data_path(), mass_spec_config))
            np.savez_compressed(os.path.join(input.data_path(), f'mz_{input.target_mz()}.npz'),eic.get())
        if input.plot_eic():
            xy_spots = np.load(os.path.join(input.data_path(), mass_spec_config.xy_spots_filename))['arr_0']

            im = pd.DataFrame(data=xy_spots, columns=['x', 'y'])
            im['z'] = eic.get()[:,1]
            z95 = np.nanpercentile(im['z'], 95)
            z5 = np.nanpercentile(im['z'], 5)
            fig = px.imshow(
                im.pivot(
                    index='y',
                    columns='x',
                    values='z',
                ).replace(np.nan, None), #https://github.com/plotly/plotly.py/issues/3470
                color_continuous_scale='viridis',
                zmin=z5,
                zmax=z95,
            ).update_layout(showlegend=False)
            fig.update_coloraxes(showscale=False)
            eic_fig_to_display.set(fig)


    @render_widget
    def kde_mz_plot():
        fig = kde_fig_to_display.get()
        if fig:
            return fig
        return None

    @render_widget
    def eic_plot():
        fig = eic_fig_to_display.get()
        if fig:
            return fig
        return None


def main():
    app = App(app_ui, server)

    def start_server():
        run_app(app, port=61235)

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