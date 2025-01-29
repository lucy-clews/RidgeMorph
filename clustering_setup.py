import pandas as pd
import pickle
import glob
import os
import umap
from sklearn import datasets, decomposition
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.colors
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
import umap.plot
from astropy.io import fits
from astropy import table
from astropy.table import Table
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
sns.set_style("white")
from sklearn.metrics import mean_squared_error


def plot_embedding(
    umap_object,
    labels=None,
    values=None,
    hover_data=None,
    tools=None,
    theme=None,
    cmap="Blues",
    color_key=None,
    color_key_cmap="PuRd",
    background="white",
    width=800,
    height=800,
    point_size=None,
    subset_points=None,
    interactive_text_search=False,
    interactive_text_search_columns=None,
    interactive_text_search_alpha_contrast=0.95,
    interactive_sample_plot=False,
    alpha=None,
):
    """Create an interactive bokeh plot of a UMAP embedding.
    While static plots are useful, sometimes a plot that
    supports interactive zooming, and hover tooltips for
    individual points is much more desirable. This function
    provides a simple interface for creating such plots. The
    result is a bokeh plot that will be displayed in a notebook.

    Note that more complex tooltips etc. will require custom
    code -- this is merely meant to provide fast and easy
    access to interactive plotting.

    Parameters
    ----------
    umap_object: trained UMAP object
        A trained UMAP object that has a 2D embedding.

    labels: array, shape (n_samples,) (optional, default None)
        An array of labels (assumed integer or categorical),
        one for each data sample.
        This will be used for coloring the points in
        the plot according to their label. Note that
        this option is mutually exclusive to the ``values``
        option.

    values: array, shape (n_samples,) (optional, default None)
        An array of values (assumed float or continuous),
        one for each sample.
        This will be used for coloring the points in
        the plot according to a colorscale associated
        to the total range of values. Note that this
        option is mutually exclusive to the ``labels``
        option.

    hover_data: DataFrame, shape (n_samples, n_tooltip_features)
    (optional, default None)
        A dataframe of tooltip data. Each column of the dataframe
        should be a Series of length ``n_samples`` providing a value
        for each data point. Column names will be used for
        identifying information within the tooltip.

    tools: List (optional, default None),
        Defines the tools to be configured for interactive plots.
        The list can be mixed type of string and tools objects defined by
        Bokeh like HoverTool. Default tool list Bokeh uses is
        ["pan","wheel_zoom","box_zoom","save","reset","help",].
        When tools are specified, and includes hovertool, automatic tooltip
        based on hover_data is not created.

    theme: string (optional, default None)
        A color theme to use for plotting. A small set of
        predefined themes are provided which have relatively
        good aesthetics. Available themes are:
           * 'blue'
           * 'red'
           * 'green'
           * 'inferno'
           * 'fire'
           * 'viridis'
           * 'darkblue'
           * 'darkred'
           * 'darkgreen'

    cmap: string (optional, default 'Blues')
        The name of a matplotlib colormap to use for coloring
        or shading points. If no labels or values are passed
        this will be used for shading points according to
        density (largely only of relevance for very large
        datasets). If values are passed this will be used for
        shading according the value. Note that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.

    color_key: dict or array, shape (n_categories) (optional, default None)
        A way to assign colors to categoricals. This can either be
        an explicit dict mapping labels to colors (as strings of form
        '#RRGGBB'), or an array like object providing one color for
        each distinct category being provided in ``labels``. Either
        way this mapping will be used to color points according to
        the label. Note that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.

    color_key_cmap: string (optional, default 'Spectral')
        The name of a matplotlib colormap to use for categorical coloring.
        If an explicit ``color_key`` is not given a color mapping for
        categories can be generated from the label list and selecting
        a matching list of colors from the given colormap. Note
        that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.

    background: string (optional, default 'white')
        The color of the background. Usually this will be either
        'white' or 'black', but any color name will work. Ideally
        one wants to match this appropriately to the colors being
        used for points etc. This is one of the things that themes
        handle for you. Note that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.

    width: int (optional, default 800)
        The desired width of the plot in pixels.

    height: int (optional, default 800)
        The desired height of the plot in pixels

    point_size: int (optional, default None)
        The size of each point marker

    subset_points: array, shape (n_samples,) (optional, default None)
        A way to select a subset of points based on an array of boolean
        values.

    interactive_text_search: bool (optional, default False)
        Whether to include a text search widget above the interactive plot

    interactive_text_search_columns: list (optional, default None)
        Columns of data source to search. Searches labels and hover_data by default.

    interactive_text_search_alpha_contrast: float (optional, default 0.95)
        Alpha value for points matching text search. Alpha value for points
        not matching text search will be 1 - interactive_text_search_alpha_contrast

    alpha: float (optional, default: None)
        The alpha blending value, between 0 (transparent) and 1 (opaque).

    Returns
    -------

    """
    if theme is not None:
        cmap = _themes[theme]["cmap"]
        color_key_cmap = _themes[theme]["color_key_cmap"]
        background = _themes[theme]["background"]

    if labels is not None and values is not None:
        raise ValueError(
            "Conflicting options; only one of labels or values should be set"
        )

    if alpha is not None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be between 0 and 1 inclusive")

    points = _get_embedding(umap_object)
    if subset_points is not None:
        if len(subset_points) != points.shape[0]:
            raise ValueError(
                "Size of subset points ({}) does not match number of input points ({})".format(
                    len(subset_points), points.shape[0]
                )
            )
        points = points[subset_points]

    if points.shape[1] != 2:
        raise ValueError("Plotting is currently only implemented for 2D embeddings")

    if point_size is None:
        point_size = 100.0 / np.sqrt(points.shape[0])

    data = pd.DataFrame(_get_embedding(umap_object), columns=("x", "y"))

    if labels is not None:
        data["label"] = np.asarray(labels)

        if color_key is None:
            unique_labels = np.unique(labels)
            num_labels = unique_labels.shape[0]
            color_key = _to_hex(
                plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels))
            )

        if isinstance(color_key, dict):
            data["color"] = pd.Series(labels).map(color_key)
        else:
            unique_labels = np.unique(labels)
            if len(color_key) < unique_labels.shape[0]:
                raise ValueError(
                    "Color key must have enough colors for the number of labels"
                )

            new_color_key = {k: color_key[i] for i, k in enumerate(unique_labels)}
            data["color"] = pd.Series(labels).map(new_color_key)

        colors = "color"

    elif values is not None:
        data["value"] = np.asarray(values)
        palette = _to_hex(plt.get_cmap(cmap)(np.linspace(0, 1, 256)))
        colors = btr.linear_cmap(
            "value", palette, low=np.min(values), high=np.max(values)
        )

    else:
        colors = matplotlib.colors.rgb2hex(plt.get_cmap(cmap)(0.5))

    if subset_points is not None:
        data = data[subset_points]
        if hover_data is not None:
            hover_data = hover_data[subset_points]

    if points.shape[0] <= width * height // 10:
        tooltips = None
        tooltip_needed = True
        if hover_data is not None:
            tooltip_dict = {}
            for col_name in hover_data:
                data[col_name] = hover_data[col_name]
                tooltip_dict[col_name] = "@{" + col_name + "}"
            tooltips = list(tooltip_dict.items())

            if tools is not None:
                for _tool in tools:
                    if _tool.__class__.__name__ == "HoverTool":
                        tooltip_needed = False
                        break

        if alpha is not None:
            data["alpha"] = alpha
        else:
            data["alpha"] = 1

        # bpl.output_notebook(hide_banner=True) # this doesn't work for non-notebook use
        data_source = bpl.ColumnDataSource(data)

        plot = bpl.figure(
            width=width,
            height=height,
            tooltips=None if not tooltip_needed else tooltips,
            tools=tools if tools is not None else "pan,wheel_zoom,box_zoom,save,reset,help",
            background_fill_color=background, output_backend="webgl"
        )
        plot.circle(
            x="x",
            y="y",
            source=data_source,
            color=colors,
            size=point_size,
            alpha="alpha",
        )

        plot.grid.visible = False
        plot.axis.visible = False
        

        get_labels = CustomJS(args=dict(s=data_source), 
            code="""
                 console.log('Running fetch indices now.');
                 var selected_labels = s.selected.indices;
                 var kernel = IPython.notebook.kernel;
                 kernel.execute("selected_labels = " + selected_labels)
                 """)

        plot.js_on_event('selectiongeometry', get_labels)

###################################

        if interactive_sample_plot:
            line_datasource =  bpl.ColumnDataSource(all_data.rename(columns={c:str(c) for c in all_data.columns}))

            # print("line_datasource instantiated")
            # curv_data = notna_data.iloc[:, :200].T
            # curv_data = curv_data.rename(columns={c:str(c) for c in curv_data.columns})
            # curv_datasource = bpl.ColumnDataSource(dict(xs=[curv_data.index.to_list() for _ in range(curv_data.shape[1])], 
            #                                           ys=[curv_data[str(i)] for i in range(curv_data.shape[1])], alpha=np.zeros(curv_data.shape[1])))
            #curvp_datasource = bpl.ColumnDataSource(dict(xs=[], 
             #                                         ys=[]))
            #code to actually make the figure analogous to plt.figure()
            #curv_plot = bpl.figure(width=400, height=400)
            # analgoust to plt.plot
            #curv_ml = MultiLine(xs="xs", ys="ys")
            #curv_plot.add_glyph(curvp_datasource, curv_ml)

            # sb_data = notna_data.iloc[:, 200:400].T
            # sb_data = sb_data.rename(columns={c:str(c) for c in sb_data.columns})
            # sb_datasource = bpl.ColumnDataSource(dict(xs=[sb_data.index.to_list() for _ in range(sb_data.shape[1])], 
            #                                           ys=[sb_data[str(i)] for i in range(sb_data.shape[1])], alpha=np.zeros(sb_data.shape[1])))
            sbp_datasource = bpl.ColumnDataSource(dict(xs=[], 
                                                      ys=[]))
            sb_plot = bpl.figure(width=400, height=400)
            sb_ml = MultiLine(xs="xs", ys="ys")
            sb_plot.add_glyph(sbp_datasource, sb_ml)

            #plot = row(plot, column(curv_plot,sb_plot))
            plot = row(plot, sb_plot)

            callback = CustomJS(args=dict(
                        s=data_source,
                        line_ds=line_datasource,
                        #p_curv=curvp_datasource,
                        sp_sb=sbp_datasource,
                    ),
                    code="""
                    var data_s = s.data;
                    const line_data = line_ds.data;
                    var inds = cb_obj.indices; 

                    sp_sb.data.xs = [];
                    sp_sb.data.ys = [];
                    //sp_curv.data.xs = [];
                    //sp_curv.data.ys = [];
                    
                    for (let j=0; j < inds.length; j++ ){
                        const ind=inds[j];
                        const shared_x = [];
                      //  const curv_y = [];
                        const sb_y = [];
                        for (let i = 0; i < 200; i++){
                        //    const curv_index = ''+i;
                            const sb_index = ''+(i+0);
                            shared_x.push(i);
                          //  curv_y.push(line_data[curv_index][ind]);
                            sb_y.push(line_data[sb_index][ind]);
                            }
                        sp_sb.data.xs.push(shared_x);
                       // sp_curv.data.xs.push(shared_x);
                        //sp_curv.data.ys.push(curv_y);
                        sp_sb.data.ys.push(sb_y);
                    }

                    //sp_curv.change.emit();
                    sp_sb.change.emit();

                """,)
            
            data_source.selected.js_on_change('indices', callback)
#####################################

        if interactive_text_search:
            text_input = TextInput(value="", title="Search:")

            if interactive_text_search_columns is None:
                interactive_text_search_columns = []
                if hover_data is not None:
                    interactive_text_search_columns.extend(hover_data.columns)
                if labels is not None:
                    interactive_text_search_columns.append("label")

            if len(interactive_text_search_columns) == 0:
                warn(
                    "interactive_text_search_columns set to True, but no hover_data or labels provided."
                    "Please provide hover_data or labels to use interactive text search."
                )

            else:
                callback = CustomJS(
                    args=dict(
                        source=data_source,
                        matching_alpha=interactive_text_search_alpha_contrast,
                        non_matching_alpha=1 - interactive_text_search_alpha_contrast,
                        search_columns=interactive_text_search_columns,
                    ),
                    code="""

                    var data = source.data;
                    var text_search = cb_obj.value;
                    
                    var search_columns_dict = {}
                    for (var col in search_columns){
                        search_columns_dict[col] = search_columns[col]
                    }
                    
                    // Loop over columns and values
                    // If there is no match for any column for a given row, change the alpha value
                    var string_match = false;
                    for (var i = 0; i < data.x.length; i++) {
                        string_match = false
                        for (var j in search_columns_dict) {
                            if (String(data[search_columns_dict[j]][i]).includes(text_search) ) {
                                string_match = true
                            }
                        }
                        if (string_match){
                            data['alpha'][i] = matching_alpha
                        }else{
                            data['alpha'][i] = non_matching_alpha
                        }
                    }
                    source.change.emit();
                """,
                )

                text_input.js_on_change("value", callback)

                plot = column(text_input, plot)

        # bpl.show(plot)
    else:
        if hover_data is not None:
            warn(
                "Too many points for hover data -- tooltips will not"
                "be displayed. Sorry; try subsampling your data."
            )
        if interactive_text_search:
            warn(
                "Too many points for text search." "Sorry; try subsampling your data."
            )
        if alpha is not None:
            warn("Alpha parameter will not be applied on holoviews plots")
        hv.extension("bokeh")
        hv.output(size=300)
        hv.opts.defaults(hv.opts.RGB(bgcolor=background, xaxis=None, yaxis=None))

        if labels is not None:
            point_plot = hv.Points(data, kdims=["x", "y"])
            plot = hd.datashade(
                point_plot,
                aggregator=ds.count_cat("color"),
                color_key=color_key,
                cmap=plt.get_cmap(cmap),
                width=width,
                height=height,
            )
        elif values is not None:
            min_val = data.values.min()
            val_range = data.values.max() - min_val
            data["val_cat"] = pd.Categorical(
                (data.values - min_val) // (val_range // 256)
            )
            point_plot = hv.Points(data, kdims=["x", "y"], vdims=["val_cat"])
            plot = hd.datashade(
                point_plot,
                aggregator=ds.count_cat("val_cat"),
                cmap=plt.get_cmap(cmap),
                width=width,
                height=height,
            )
        else:
            point_plot = hv.Points(data, kdims=["x", "y"])
            plot = hd.datashade(
                point_plot,
                aggregator=ds.count(),
                cmap=plt.get_cmap(cmap),
                width=width,
                height=height,
            )

    
    return plot, sbp_datasource, data_source

