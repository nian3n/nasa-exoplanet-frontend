import streamlit as st
PAGE_COLORS = {
    "forest":    "#284139",  # deep green
    "sage":      "#809076",  # muted green
    "sand":      "#F8D794",  # light accent 
    "ochre":     "#886830",  # warm brown
    "deep_teal": "#22393C",  # dark teal
}

# Use the light 'sand' tone as the mist block color (for uploader, etc.)
MIST_HEX = PAGE_COLORS["sand"]

def _hex_to_rgb(hx: str):
    hx = hx.lstrip("#")
    return tuple(int(hx[i:i+2], 16) for i in (0, 2, 4))

def mix_rgba(hex_color: str, alpha: float = 0.6, bias: float = 0.0) -> str:
    """Return rgba(r,g,b,a) with an optional bias to brighten (+) or darken (-) the color."""
    r, g, b = _hex_to_rgb(hex_color)
    if bias != 0.0:
        r = max(0, min(255, int(r * (1 + bias))))
        g = max(0, min(255, int(g * (1 + bias))))
        b = max(0, min(255, int(b * (1 + bias))))
    return f"rgba({r},{g},{b},{alpha})"

def apply_theme(page_key: str, nebula_path: str = "assets/Nebula.png"):
    base_hex = PAGE_COLORS.get(page_key, PAGE_COLORS["forest"])

    # Tuned opacities
    panel_rgba   = mix_rgba(base_hex, alpha=0.55, bias=0.08)
    mist_rgba    = mix_rgba(MIST_HEX,  alpha=0.85, bias=0.00)  
    sidebar_rgba = mix_rgba(base_hex, alpha=0.85, bias=-0.15)

    # A subtle starfield + gradient overlay to avoid monotone look
    gradient_overlay = (
        "radial-gradient(1200px 600px at 10% 20%, rgba(255,255,255,0.06), rgba(0,0,0,0) 60%),"
        "radial-gradient(900px 500px at 80% 30%, rgba(255,255,255,0.05), rgba(0,0,0,0) 55%),"
        "linear-gradient(180deg, rgba(0,0,30,0.45) 0%, rgba(0,0,0,0.75) 100%)"
    )

    st.markdown(f"""
    <style>
      /* Base app background (solid color, acts as a fallback under the hero) */
      .stApp {{
        background: {base_hex} !important;
      }}

      /* Default white captions/text */
      h1,h2,h3,h4,h5,p,li,span,small,code,label {{
        color: #fff !important;
      }}

      /* Glass panels */
      .glass {{
        background: {panel_rgba};
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 16px;
        padding: 18px 22px;
        margin: 12px 0;
        box-shadow: 0 8px 28px rgba(0,0,0,0.35);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
      }}

      /* Sidebar*/
      section[data-testid="stSidebar"] > div {{
        background: {sidebar_rgba} !important;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-right: 1px solid rgba(0,0,0,0.35);
      }}

      /* File uploader Block */
        .stFileUploader > div:first-child {{
            background: {mist_rgba} !important;
            border-radius: 14px !important;
            border: 1px solid rgba(255,255,255,0.25) !important;
            padding: 12px;
            text-align: center;
        }}

        .stFileUploader label {{
            color: #fff !important;
            font-size: clamp(14px, 2vw, 20px);
            font-weight: 600;
            text-align: center;
            text-shadow: 0 0 6px rgba(0,0,0,0.45);
        }}

        .stFileUploader span,
        .stFileUploader p {{
            color: #c2dde4 !important;              /* mist-tone text */
            font-size: clamp(14px, 2vw, 20px);   /* scale like subtitle */
            font-weight: 500;
            text-align: center;
            text-shadow: 0 0 6px rgba(0,0,0,0.45); /* glow for contrast */
        }}


      /* HEADER */
      .exo-hero {{
        width: 100%;
        min-height: 36vh;
        max-height: 48vh;
        border-radius: 18px;
        overflow: hidden;
        position: relative;
        margin: 6px 0 20px 0;
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 12px 38px rgba(0,0,0,0.45);
        background-image:
          {gradient_overlay},
          url("{nebula_path}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed; /* keeps nebula steady while scrolling */
      }}

      .exo-hero-inner {{
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 28px 24px;
      }}

      .exo-title {{
        font-size: clamp(28px, 5vw, 56px);
        font-weight: 800;
        letter-spacing: 1.5px;
        margin: 0;
        text-shadow: 0 0 12px rgba(255,255,255,0.5), 0 0 28px rgba(97,218,251,0.45);
      }}

      .exo-subtitle {{
        margin-top: 8px;
        font-size: clamp(14px, 2.2vw, 22px);
        color: #c2dde4;
        text-shadow: 0 0 14px rgba(0,0,0,0.65);
      }}

      /* tiny planets (non-monotone spark) */
      .exo-hero::before,
      .exo-hero::after {{
        content: "";
        position: absolute;
        border-radius: 50%;
        filter: blur(0.5px);
        opacity: 0.9;
      }}
      .exo-hero::before {{
        width: 22px; height: 22px; top: 18%; left: 12%;
        background: radial-gradient(circle at 35% 35%, #ffd2a6 0%, #d97a00 45%, rgba(0,0,0,0) 60%);
        box-shadow: 0 0 10px rgba(255,210,166,0.6);
      }}
      .exo-hero::after {{
        width: 14px; height: 14px; bottom: 16%; right: 18%;
        background: radial-gradient(circle at 35% 35%, #b9d6ff 0%, #2e7dd6 50%, rgba(0,0,0,0) 65%);
        box-shadow: 0 0 8px rgba(185,214,255,0.6);
      }}
    </style>
    """, unsafe_allow_html=True)

def header(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="exo-hero">
          <div class="exo-hero-inner">
            <div>
              <h1 class="exo-title">{title}</h1>
              {f'<div class="exo-subtitle">{subtitle}</div>' if subtitle else ''}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
