
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm

def plot_mapa_estados_continuo(
    df,
    geo_path,
    value_col="perc_cotas_negras",
    modo="ESTADO",
    col_uf_geo="id",
    col_uf_df="uf",
    bins=None,
    labels=None,
    colors=None,
    right=True,
    title=None,
    save_path=None,
    figsize=(10, 10),
    edgecolor="black",
    linewidth=0.7,
    missing_color="#eeeeee",
    show=True
):
    """
    Fluxo:
    1) Agrega por UF (média)
    2) Arredonda valores para 2 casas decimais (proporção)
    3) Cria versão half-up em passos de 1% para classificação
    4) Plota mapa com classes discretas
    """
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bins, ncolors=cmap.N, clip=True)

    if bins is None or labels is None or cmap is None or norm is None:
        raise ValueError("Passe bins, labels, cmap e norm (definidos fora da função).")

    # -------- GEO --------
    g = gpd.read_file(geo_path).copy()
    g["uf"] = g[col_uf_geo].astype(str).str.strip().str.upper()

    # -------- BASE --------
    d = df.copy()
    d[col_uf_df] = d[col_uf_df].astype(str).str.strip().str.upper()

    if modo.upper() == "ESTADO":
        d = d[d["tipo_ente"].astype(str).str.upper().eq("ESTADO")].copy()
    elif modo.upper() == "CAPITAL":
        d = d[d["tipo_ente"].astype(str).str.upper().eq("CAPITAL")].copy()
    else:
        raise ValueError("modo deve ser 'ESTADO' ou 'CAPITAL'")

    base = (
        d.groupby(col_uf_df, as_index=False)[value_col]
         .mean()
         .rename(columns={col_uf_df: "uf"})
    )

    # -------- MERGE --------
    m = g.merge(base, on="uf", how="left")

    # =====================================================
    # 1) ARREDONDAMENTO VISUAL: 2 CASAS DECIMAIS (0–1)
    # =====================================================
    vc_r2 = value_col + "_r2"
    m[vc_r2] = m[value_col].round(2)

    # =====================================================
    # 2) ARREDONDAMENTO HALF-UP PARA CLASSIFICAÇÃO (1%)
    # =====================================================
    vc_p1 = value_col + "_p1"
    m[vc_p1] = np.floor(m[vc_r2].astype(float) * 100 + 0.5) / 100

    # -------- CLASSE (DEBUG / LEGENDA) --------
    m["classe"] = pd.cut(
        m[vc_p1],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=right
    )

    # -------- PLOT --------
    fig, ax = plt.subplots(figsize=figsize)

    m.plot(
        column=vc_p1,
        cmap=cmap,
        norm=norm,
        linewidth=linewidth,
        edgecolor=edgecolor,
        ax=ax,
        missing_kwds={"color": missing_color, "label": "Sem dado"},
    )

    ax.set_axis_off()
    # ax.set_title(title or f"Brasil — {value_col} ({modo.capitalize()})")

    # -------- COLORBAR DISCRETA --------
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)

    midpoints = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    cbar.set_ticks(midpoints)
    cbar.set_ticklabels(labels)
    # cbar.set_label(value_col)

    plt.tight_layout()

    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{save_path}.svg", bbox_inches="tight")
    if show is True:
        plt.show()
        plt.close(fig)
    else: 
        return None