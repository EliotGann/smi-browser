"""ScanCollection — holds processed results for comparison and sweeps."""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ScanCollection:
    """
    Manages a set of processed CombinedReductionResults for comparison
    and parameter-sweep analysis.

    Within-scan variation (e.g. waxs_arc angle per frame) is already
    handled by reduce_smi_combined.  Between-scan variation (sample,
    temperature, etc.) is tracked here.

    Usage
    -----
        coll = ScanCollection()
        coll.add(result, metadata, params)
        coll.varying_parameters()      # which metadata fields differ
        coll.iq_comparison_figure()    # matplotlib overlay
        ds = coll.stack_iq("sample")   # xr.Dataset along a new dim
    """

    _PALETTE = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    def __init__(self):
        self._results: dict[str, Any] = {}           # uid -> CombinedReductionResult
        self._metadata: dict[str, dict] = {}         # uid -> enhanced_summary dict
        self._processing: dict[str, dict] = {}       # uid -> processing kwargs
        self._colors: dict[str, str] = {}            # uid -> hex color
        self._color_idx = 0
        self._primary_dfs: dict[str, pd.DataFrame] = {}   # uid -> primary scalars
        self._baseline_dfs: dict[str, pd.DataFrame] = {}  # uid -> baseline scalars
        self._raw_metadata: dict[str, dict] = {}          # uid -> full tiled metadata

    @property
    def uids(self) -> list[str]:
        return list(self._results.keys())

    def __len__(self):
        return len(self._results)

    def __contains__(self, uid: str):
        return uid in self._results

    def add(self, result, metadata: dict, params: dict | None = None,
             primary_df: pd.DataFrame | None = None,
             baseline_df: pd.DataFrame | None = None,
             raw_metadata: dict | None = None):
        """Add a processed scan to the collection.

        Parameters
        ----------
        primary_df : DataFrame, optional
            Primary stream scalar data (for label columns / export).
        baseline_df : DataFrame, optional
            Baseline stream scalar data (for label columns / export).
        raw_metadata : dict, optional
            Full tiled metadata dict (for export).
        """
        self._results[result.uid] = result
        self._metadata[result.uid] = metadata
        if params:
            self._processing[result.uid] = params
        if primary_df is not None:
            self._primary_dfs[result.uid] = primary_df
        if baseline_df is not None:
            self._baseline_dfs[result.uid] = baseline_df
        if raw_metadata is not None:
            self._raw_metadata[result.uid] = raw_metadata
        if result.uid not in self._colors:
            self._colors[result.uid] = self._PALETTE[
                self._color_idx % len(self._PALETTE)
            ]
            self._color_idx += 1

    def remove(self, uid: str):
        self._results.pop(uid, None)
        self._metadata.pop(uid, None)
        self._processing.pop(uid, None)
        self._colors.pop(uid, None)
        self._primary_dfs.pop(uid, None)
        self._baseline_dfs.pop(uid, None)
        self._raw_metadata.pop(uid, None)

    def get_color(self, uid: str) -> str:
        return self._colors.get(uid, "#888888")

    def set_color(self, uid: str, color: str) -> None:
        if uid in self._results:
            self._colors[uid] = color

    def get_result(self, uid: str):
        return self._results.get(uid)

    def available_label_columns(self) -> list[str]:
        """Return sorted list of numeric column names from stored primary/baseline data."""
        cols: set[str] = set()
        for df in list(self._primary_dfs.values()):
            cols.update(c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]))
        for df in list(self._baseline_dfs.values()):
            cols.update(
                f"baseline:{c}" for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])
            )
        return sorted(cols)

    def get_label_value(self, uid: str, column: str) -> str:
        """Get a representative label value for a scan from primary or baseline."""
        if column.startswith("baseline:"):
            field = column[len("baseline:"):]
            df = self._baseline_dfs.get(uid)
            if df is not None and field in df.columns:
                vals = df[field].dropna()
                if len(vals):
                    return f"{vals.iloc[0]:.6g}"
        else:
            df = self._primary_dfs.get(uid)
            if df is not None and column in df.columns:
                vals = df[column].dropna()
                if len(vals):
                    # Use median as representative value for multi-frame scans
                    return f"{vals.median():.6g}"
        return "?"

    def get_primary_df(self, uid: str) -> pd.DataFrame | None:
        return self._primary_dfs.get(uid)

    def get_baseline_df(self, uid: str) -> pd.DataFrame | None:
        return self._baseline_dfs.get(uid)

    def get_raw_metadata(self, uid: str) -> dict | None:
        return self._raw_metadata.get(uid)

    def summary_table(self, label_column: str | None = None) -> pd.DataFrame:
        """DataFrame summary of all scans in the collection.

        Parameters
        ----------
        label_column : str, optional
            A primary or baseline field name (baseline fields prefixed with
            ``baseline:``) to include as an extra column in the table.
        """
        rows = []
        for uid in self._results:
            res = self._results[uid]
            meta = self._metadata.get(uid, {})
            timing = res.timing or {}
            det_list = meta.get("detector_list")
            if det_list and isinstance(det_list, list):
                detectors = ", ".join(det_list)
            else:
                detectors = meta.get("detectors", "?")
            row = {
                "color":     self._colors.get(uid, "#888888"),
                "uid_short": uid[:8],
                "sample":    meta.get("sample_name", "?"),
                "plan":      meta.get("plan_name", "?"),
                "detectors": detectors,
                "geometry":  res.geometry,
                "total_s":   f"{sum(timing.values()):.1f}" if timing else "?",
                "uid":       uid,
            }
            if label_column:
                row["label_val"] = self.get_label_value(uid, label_column)
            rows.append(row)
        cols = ["color", "uid_short", "sample", "plan", "detectors", "geometry", "total_s", "uid"]
        if label_column:
            cols.insert(3, "label_val")
        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

    def varying_parameters(self) -> dict[str, list]:
        """
        Identify metadata fields that differ across scans in the collection.
        Skips fields that are inherently unique per-scan (uid, scan_id, time).
        """
        if len(self._metadata) < 2:
            return {}
        all_metas = list(self._metadata.values())
        all_keys: set[str] = set()
        for m in all_metas:
            all_keys.update(m.keys())

        skip = {
            "uid", "scan_id", "time", "exit_status", "streams",
            "detector_list", "has_saxs", "has_waxs", "n_steps",
        }
        varying = {}
        for key in sorted(all_keys - skip):
            vals = [str(m.get(key, "")) for m in all_metas]
            if len(set(vals)) > 1:
                varying[key] = vals
        return varying

    def iq_comparison_figure(self, figsize=(9, 5)):
        """Matplotlib figure overlaying I(q) from every scan."""
        if not self._results:
            return None
        fig, ax = plt.subplots(figsize=figsize)
        for uid, res in self._results.items():
            meta = self._metadata.get(uid, {})
            label = f"{uid[:8]} — {meta.get('sample_name', '?')}"
            iq = res.merged_iq
            q = iq["q"].values
            I = iq["I"].values
            mask = np.isfinite(I) & (I > 0)
            if mask.any():
                ax.plot(q[mask], I[mask], linewidth=0.8, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("q (nm⁻¹)")
        ax.set_ylabel("I(q)")
        ax.set_title("Processed Collection — I(q) Comparison")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        return fig

    def iq_comparison_bokeh(self, uids: list[str] | None = None,
                            label_column: str | None = None):
        """Bokeh figure overlaying I(q) for selected uids (or all).

        Uses stored per-scan colours (matching the table swatch) and
        omits the legend — the table serves as the interactive legend.
        Hovering highlights the nearest curve and dims the others.

        Parameters
        ----------
        label_column : str, optional
            If given, the chosen primary/baseline value is appended to
            the hover tooltip label for each curve.
        """
        from bokeh.events import MouseLeave
        from bokeh.models import CustomJS, HoverTool
        from bokeh.models.glyphs import Line as BkLine
        from bokeh.plotting import figure as bk_figure

        subset = uids if uids else list(self._results.keys())
        subset = [u for u in subset if u in self._results]
        if not subset:
            return None

        p = bk_figure(
            title="Processed Collection — I(q) Comparison",
            width=1000, height=500,
            x_axis_type="log", y_axis_type="log",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            sizing_mode="stretch_both",
        )

        renderers = []
        for uid in subset:
            res = self._results[uid]
            meta = self._metadata.get(uid, {})
            label_parts = [uid[:8], meta.get('sample_name', '?')]
            if label_column:
                label_parts.append(f"{label_column}={self.get_label_value(uid, label_column)}")
            label = " — ".join(label_parts)
            color = self._colors.get(uid, "#888888")
            iq = res.merged_iq
            if iq is None:
                continue
            q = iq["q"].values
            I = iq["I"].values
            mask = np.isfinite(I) & (I > 0)
            if mask.any():
                r = p.line(
                    q[mask], I[mask],
                    line_width=1.2, line_alpha=0.4,
                    color=color, name=label,
                )
                # Set hover glyph properties individually to avoid
                # Bokeh 3.x Value() wrapper serialization issues.
                hover_glyph = BkLine()
                hover_glyph.line_color = color
                hover_glyph.line_alpha = 1.0
                hover_glyph.line_width = 3.5
                r.hover_glyph = hover_glyph
                renderers.append(r)

        p.xaxis.axis_label = "q (nm⁻¹)"
        p.yaxis.axis_label = "I(q)"
        if p.legend:
            p.legend.visible = False

        if renderers:
            hover_cb = CustomJS(args=dict(renderers=renderers), code="""
                const r = cb_obj.renderers;
                const inspected = r.filter(
                    rend => rend.inspected && rend.inspected.indices.length > 0
                );
                if (inspected.length > 0) {
                    for (const rend of renderers) {
                        if (inspected.includes(rend)) {
                            rend.glyph.line_alpha = 1.0;
                            rend.glyph.line_width = 3.5;
                        } else {
                            rend.glyph.line_alpha = 0.08;
                            rend.glyph.line_width = 0.7;
                        }
                    }
                }
            """)
            reset_cb = CustomJS(args=dict(renderers=renderers), code="""
                for (const r of renderers) {
                    r.glyph.line_alpha = 0.4;
                    r.glyph.line_width = 1.2;
                }
            """)
            hover = HoverTool(
                renderers=renderers,
                tooltips=[
                    ("scan", "$name"),
                    ("q", "$x{0.000}"),
                    ("I", "$y{0.00e+0}"),
                ],
                line_policy="nearest",
                mode="mouse",
                callback=hover_cb,
            )
            p.add_tools(hover)
            p.js_on_event(MouseLeave, reset_cb)

        return p

    def stack_iq(self, dim_name: str = "scan", dim_values=None):
        """
        Stack merged I(q) datasets into a single xr.Dataset along a new dim.
        If dim_values is given, use those as the coordinate; otherwise use
        short UID + sample_name labels.
        """
        import xarray as xr
        if not self._results:
            return xr.Dataset()
        datasets = []
        labels = []
        for i, (uid, res) in enumerate(self._results.items()):
            if dim_values is not None and i < len(dim_values):
                labels.append(dim_values[i])
            else:
                meta = self._metadata.get(uid, {})
                labels.append(f"{uid[:8]}_{meta.get('sample_name', '?')}")
            datasets.append(res.merged_iq)
        return xr.concat(datasets, dim=pd.Index(labels, name=dim_name))
