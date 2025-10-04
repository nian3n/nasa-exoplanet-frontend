import streamlit as st
PAGE_COLORS = {
    "forest":    "#284139",  # deep green
    "sage":      "#809076",  # muted green
    "sand":      "#18066C",  # dark blue
    "ochre":     "#886830",  # warm brown
    "deep_teal": "#110515",  # dark black
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

def apply_theme(page_key: str, nebula_path: str = "assets/Nebula.png", uploader_caption_hex: str = "#111111",accent_hex: str | None = "#0f766e"):
    base_hex = PAGE_COLORS.get(page_key, PAGE_COLORS["forest"])
    accent_hex = accent_hex or "#0f766e"

    # Tuned opacities
    panel_rgba   = mix_rgba(base_hex, alpha=0.55, bias=0.08)
    mist_rgba    = mix_rgba(MIST_HEX,  alpha=0.85, bias=0.00)  
    sidebar_rgba = mix_rgba(base_hex, alpha=0.85, bias=-0.15)

    accent_glow  = mix_rgba(accent_hex, alpha=0.20)
    accent_faint = mix_rgba(accent_hex, alpha=0.25)

    gradient_overlay = (
        "radial-gradient(1200px 600px at 10% 20%, rgba(255,255,255,0.06), rgba(0,0,0,0) 60%),"
        "radial-gradient(900px 500px at 80% 30%, rgba(255,255,255,0.05), rgba(0,0,0,0) 55%),"
        "linear-gradient(180deg, rgba(0,0,30,0.45) 0%, rgba(0,0,0,0.75) 100%)"
    )
    uploader_bg    = lighten_with_white(base_hex, mix=0.14, alpha=0.88)  # slightly lighter than bg
    uploader_border= lighten_with_white(base_hex, mix=0.32, alpha=0.95)  # a bit brighter for edge


    st.markdown(f"""
    <style>
        :root {{
            --exo-uploader-caption: {uploader_caption_hex};
            --primary-color: {accent_hex};
        }}

        /* File uploader: white box + per-page caption color */
        .stFileUploader > div:first-child {{
            background: #ffffff !important;
            border: 1px solid rgba(0,0,0,0.15) !important;
            border-radius: 0 !important;
            padding: 12px;
            text-align: center;
        box-shadow: none !important;
  }}

  /* Only uploader text uses the per-page color */
  .stFileUploader label,
  .stFileUploader span,
  .stFileUploader p,
  .stFileUploader [data-testid="stFileUploadDropzone"] * {{
      color: var(--exo-uploader-caption) !important;
      text-shadow: none !important;
  }}
        /* Page background */
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


      /* HEADER */
      .exo-hero {{
        width: 100%;
        min-height: 36vh;
        max-height: 48vh;
        border-radius: 0 !important;;
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

      /* tiny planets */
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

def lighten_with_white(hex_color: str, mix: float = 0.14, alpha: float = 0.85) -> str:
    """
    Lighten a hex color by mixing with white (0–1), return rgba().
    mix=0.14 means 14% closer to white. alpha in [0,1].
    """
    r, g, b = _hex_to_rgb(hex_color.lstrip("#"))
    r = round(r + (255 - r) * mix)
    g = round(g + (255 - g) * mix)
    b = round(b + (255 - b) * mix)
    return f"rgba({r},{g},{b},{alpha})"

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
