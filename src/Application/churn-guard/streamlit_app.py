"""churnguard - churn risk app for the mldp project.

scores one customer at a time, shows what pushed the score up or down, and
works out which retention actions would actually help. model comes from
MLDP_notebook.ipynb
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="ChurnGuard",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.pkl"


@st.cache_resource
def load_model(path):
    return joblib.load(path)


try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error("Could not find 'model.pkl'. Keep it in the same folder as this app.")
    st.stop()
except Exception as exc:
    st.error(f"Failed to load the model: {exc}")
    st.stop()

POS = list(model.classes_).index(1)

## ---------------------------------------------------------------------------
## all the styling. basically dark background with some glow behind it, panels
## are slightly see-through so the glow shows through. green = good/risk going
## down, red = bad. took a few tries to get this looking decent tbh
## ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;450;500;600;700&display=swap');

:root{
  --bg:#080a12;
  --surface:rgba(255,255,255,.030);
  --surface2:rgba(255,255,255,.055);
  --edge:rgba(255,255,255,.075);
  --edge2:rgba(255,255,255,.13);
  --ink:#f0f3fa; --ink2:#a3adc2; --ink3:#6b7590;
  --blue:#5b8cff; --violet:#a78bfa;
  --mint:#4ade80; --amber:#fbbf24; --rose:#fb7185;
  --r:18px;
}

html,body,[class*="css"]{font-family:'Inter',ui-sans-serif,system-ui,sans-serif;
  font-variant-numeric:tabular-nums;-webkit-font-smoothing:antialiased}
.stApp{
  background:
    radial-gradient(900px 520px at 8% -12%, rgba(91,140,255,.20), transparent 70%),
    radial-gradient(760px 460px at 96% -6%, rgba(167,139,250,.16), transparent 68%),
    radial-gradient(700px 700px at 50% 118%, rgba(91,140,255,.07), transparent 70%),
    var(--bg);
}
#MainMenu,footer{visibility:hidden}
header[data-testid="stHeader"]{background:transparent;height:2.2rem}
.block-container{max-width:1240px;padding:2.4rem 2rem 4rem}
div[data-testid="stVerticalBlock"]{gap:.6rem}

label{font-size:.9rem!important;color:var(--ink2)!important;font-weight:450!important}
div[data-baseweb="select"]>div{background:rgba(255,255,255,.04)!important;
  border-color:var(--edge)!important;border-radius:11px!important;font-size:.93rem}
div[role="listbox"]{background:#11141f!important;border:1px solid var(--edge2)!important}
div[data-testid="stCaptionContainer"] p{font-size:.86rem!important;color:var(--ink3)!important;line-height:1.6}
details summary{font-size:.93rem!important;font-weight:600!important;color:var(--ink2)!important}
div[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--surface);border:1px solid var(--edge)!important;
  border-radius:var(--r)!important;padding:.55rem;backdrop-filter:blur(8px)}

/* ---------- top rail ---------- */
.rail{display:flex;align-items:center;gap:.7rem;margin-bottom:3rem}
.rail .mk{width:32px;height:32px;border-radius:10px;display:grid;place-items:center;
  background:linear-gradient(135deg,var(--blue),var(--violet));color:#0a0d16;
  font-weight:700;font-size:1rem;box-shadow:0 6px 20px -6px rgba(91,140,255,.75)}
.rail .wm{font-family:'Space Grotesk';font-size:1.06rem;font-weight:600;color:var(--ink);letter-spacing:-.2px}
.rail .rt{margin-left:auto;display:flex;align-items:center;gap:.5rem}
.tag{font-size:.74rem;font-weight:500;color:var(--ink3);padding:.32rem .7rem;
  border:1px solid var(--edge);border-radius:8px;background:var(--surface)}
.tag.on{color:var(--mint);border-color:rgba(74,222,128,.3);background:rgba(74,222,128,.08);
  display:inline-flex;align-items:center;gap:.45rem}
.blip{width:7px;height:7px;border-radius:50%;background:var(--mint);display:inline-block;flex:none;
  box-shadow:0 0 8px var(--mint);animation:blip 1.8s ease-in-out infinite}
@keyframes blip{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.25;transform:scale(.82)}}

/* ---------- hero ---------- */
.hero{display:grid;grid-template-columns:1.15fr .85fr;gap:3rem;align-items:center;margin-bottom:2.6rem}
.hero h1{font-family:'Space Grotesk';font-weight:600;letter-spacing:-.035em;
  font-size:clamp(2.3rem,4.4vw,3.55rem);line-height:1.02;margin:0;color:var(--ink)}

/* ---------- pinned score, stays visible while scrolling ---------- */
.pin{position:fixed;right:1.6rem;bottom:1.6rem;z-index:9999;min-width:150px;
  background:rgba(14,18,31,.86);backdrop-filter:blur(16px) saturate(1.3);
  border:1px solid var(--edge2);border-radius:17px;padding:.9rem 1.05rem;
  box-shadow:0 26px 60px -22px rgba(0,0,0,.95),0 0 0 1px rgba(255,255,255,.03) inset}
.pin .k{font-size:.66rem;font-weight:600;letter-spacing:.11em;text-transform:uppercase;
  color:var(--ink3);display:flex;align-items:center;gap:.4rem}
.pin .v{font-family:'Space Grotesk';font-size:2.15rem;font-weight:700;letter-spacing:-.05em;
  line-height:1.05;margin-top:.15rem}
.pin .v i{font-style:normal;font-size:1rem;font-weight:600;opacity:.5;margin-left:.1rem}
.pin .t{height:4px;border-radius:99px;background:rgba(255,255,255,.09);overflow:hidden;margin-top:.55rem}
.pin .t>div{height:100%;border-radius:99px}
.pin .l{font-size:.68rem;font-weight:600;letter-spacing:.06em;margin-top:.45rem}
@media(max-width:760px){.pin{right:.8rem;bottom:.8rem;padding:.7rem .85rem;min-width:120px}
  .pin .v{font-size:1.7rem}}
.hero h1 span{background:linear-gradient(100deg,var(--blue),var(--violet));
  -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.hero p{max-width:44ch;margin:1.15rem 0 0;font-size:1.02rem;line-height:1.72;color:var(--ink2)}

.dial{background:var(--surface);border:1px solid var(--edge);border-radius:24px;
  padding:1.8rem 1.9rem;backdrop-filter:blur(10px);box-shadow:0 24px 60px -30px rgba(0,0,0,.9)}
.dial .cap{font-size:.72rem;font-weight:600;letter-spacing:.11em;text-transform:uppercase;color:var(--ink3)}
.dial .big{font-family:'Space Grotesk';font-size:4.6rem;font-weight:700;letter-spacing:-.055em;
  line-height:.94;margin:.5rem 0 .1rem}
.dial .big i{font-style:normal;font-size:1.7rem;font-weight:600;opacity:.5;margin-left:.15rem}
.dial .vd{font-size:.95rem;color:var(--ink2);margin-top:.35rem}
.rail-bar{height:8px;border-radius:99px;background:rgba(255,255,255,.07);overflow:hidden;margin:1.15rem 0 .4rem}
.rail-bar>div{height:100%;border-radius:99px}
.ticks{display:flex;justify-content:space-between;font-size:.72rem;color:var(--ink3)}
.chip{display:inline-block;padding:.3rem .8rem;border-radius:8px;font-size:.74rem;
  font-weight:600;letter-spacing:.05em}

/* ---------- metric strip ---------- */
.mets{display:grid;grid-template-columns:repeat(4,1fr);gap:.85rem;margin-bottom:2.6rem}
.met{background:var(--surface);border:1px solid var(--edge);border-radius:14px;padding:1.05rem 1.15rem}
.met .k{font-size:.71rem;font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:var(--ink3)}
.met .v{font-family:'Space Grotesk';font-size:1.62rem;font-weight:600;color:var(--ink);
  letter-spacing:-.03em;margin-top:.3rem}
.met .v em{font-style:normal;font-size:.9rem;color:var(--ink3);font-weight:500;margin-left:.2rem}

/* ---------- section heads ---------- */
.sec{display:flex;align-items:baseline;gap:.85rem;margin:0 0 1.05rem}
.sec h2{font-family:'Space Grotesk';font-size:1.12rem;font-weight:600;color:var(--ink);
  margin:0;letter-spacing:-.015em}
.sec p{margin:0;font-size:.87rem;color:var(--ink3)}

.grp{font-size:.72rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink3);margin:.15rem 0 .55rem;display:flex;align-items:center;gap:.45rem}
.grp b{width:5px;height:5px;border-radius:99px;display:inline-block}

.lead{max-width:78ch;font-size:.93rem;line-height:1.75;color:var(--ink2);margin:-.35rem 0 1.15rem}
.lead strong{color:var(--ink);font-weight:600}
.pane{background:var(--surface);border:1px solid var(--edge);border-radius:var(--r);
  padding:1.15rem 1.3rem;margin-bottom:.85rem}
.pane .ph{font-size:.72rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink3);margin-bottom:.85rem}

/* ---------- attribution ---------- */
.dv{display:flex;align-items:center;gap:.7rem;margin:.42rem 0}
.dv .t{font-size:.89rem;color:var(--ink2);flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.dv .b{height:5px;border-radius:99px;width:66px;background:rgba(255,255,255,.07);flex:none}
.dv .b>div{height:100%;border-radius:99px}

/* ---------- retention levers ---------- */
.lv{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:.9rem}
.lvc{position:relative;background:var(--surface);border:1px solid var(--edge);
  border-radius:16px;padding:1.15rem 1.2rem;overflow:hidden;transition:border-color .15s}
.lvc::after{content:"";position:absolute;inset:0 auto 0 0;width:2px;background:var(--mint);opacity:.35}
.lvc.best{border-color:rgba(74,222,128,.34);background:rgba(74,222,128,.055)}
.lvc.best::after{opacity:1}
.lvc .rk{position:absolute;top:.85rem;right:1rem;font-family:'Space Grotesk';
  font-size:.76rem;font-weight:600;color:var(--ink3)}
.lvc .ac{font-size:.93rem;font-weight:550;color:var(--ink);line-height:1.35;
  min-height:2.5rem;padding-right:1.4rem}
.lvc .mv{font-size:.82rem;color:var(--ink3);margin-top:.7rem}
.lvc .mv b{color:var(--ink2);font-weight:600}
.lvc .dl{font-family:'Space Grotesk';font-size:1.75rem;font-weight:700;
  letter-spacing:-.045em;color:var(--mint);margin-top:.15rem}
.lvc .bar{height:4px;border-radius:99px;background:rgba(255,255,255,.07);margin-top:.75rem;overflow:hidden}
.lvc .bar>div{height:100%;border-radius:99px;background:var(--mint)}
.note{font-size:.85rem;line-height:1.7;color:var(--ink3);max-width:96ch;margin-top:.9rem}
.empty{color:var(--ink3);font-size:.92rem;line-height:1.65}
.foot{font-size:.84rem;line-height:1.7;color:var(--ink3);margin-top:.6rem}

@media(max-width:900px){.hero{grid-template-columns:1fr;gap:2rem}
  .mets{grid-template-columns:repeat(2,1fr)}}
</style>
""", unsafe_allow_html=True)

## ---------------------------------------------------------------------------
## same feature engineering as section 3.2 in the notebook. kept these as
## seperate functions because the lever bit further down needs to redo them
## for every what-if it tries
## ---------------------------------------------------------------------------
SERVICE_COLS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


def derive(raw):
    d = dict(raw)
    d["num_services"] = (sum(1 for c in SERVICE_COLS if d[c] == "Yes")
                         + (1 if d["InternetService"] != "No" else 0))
    t = d["tenure"]
    d["tenure_group"] = ("New (0-12m)" if t <= 12 else
                         "Established (12-24m)" if t <= 24 else
                         "Loyal (24-48m)" if t <= 48 else
                         "Veteran (48m+)")
    return d


def encode(d):
    return pd.get_dummies(pd.DataFrame([d])).reindex(
        columns=model.feature_names_in_, fill_value=0)


def score(raw):
    return float(model.predict_proba(encode(derive(raw)))[0][POS])


## checks the inputs actually make sense before we trust the number. the widgets
## already stop out-of-range values so this is about combinations that cant
## happen in real life, plus warning when were outside what the model was
## trained on (it will still give u a number, it just wont mean much)
TRAIN_RANGES = {                     ## min/max seen in the 7,032 training rows
    "monthly": (18.25, 118.75),
    "total": (18.80, 8684.80),
    "implied": (13.78, 121.40),      ## TotalCharges / tenure
}


def validate(raw, tenure, monthly, total):
    errors, warnings = [], []

    ## impossible combos - these cant exist in the data
    net = raw["InternetService"]
    addons = {"OnlineSecurity": "Online security", "OnlineBackup": "Online backup",
              "DeviceProtection": "Device protection", "TechSupport": "Tech support",
              "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies"}
    if net == "No":
        wrong = [lbl for f, lbl in addons.items() if raw[f] == "Yes"]
        if wrong:
            errors.append(
                f"{', '.join(wrong)} cannot be active when Internet service is set to "
                f"'No'. Set Internet service, or set these to 'No internet service'.")

    if raw["PhoneService"] == "No" and raw["MultipleLines"] == "Yes":
        errors.append("Multiple lines cannot be 'Yes' when Phone service is 'No'.")

    if total < monthly:
        errors.append(
            f"Total charges (${total:,.2f}) are lower than one month's charge "
            f"(${monthly:,.2f}). A customer billed at least once cannot have paid less "
            f"than a single month.")

    ## out-of-range warnings - model still runs, but its extrapolating
    lo, hi = TRAIN_RANGES["implied"]
    implied = total / tenure if tenure else 0
    if not lo <= implied <= hi:
        warnings.append(
            f"Average monthly spend works out at ${implied:,.2f}, outside the "
            f"${lo:,.2f}–${hi:,.2f} range seen in training. The prediction is an "
            f"extrapolation and should be treated with caution.")

    lo, hi = TRAIN_RANGES["total"]
    if not lo <= total <= hi:
        warnings.append(
            f"Total charges of ${total:,.2f} fall outside the ${lo:,.0f}–${hi:,.0f} "
            f"range in the training data.")

    return errors, warnings


def band_of(p):
    if p < 30:
        return "LOW RISK", "var(--mint)", "#4ade80", "This customer looks settled."
    if p < 60:
        return "ELEVATED", "var(--amber)", "#fbbf24", "Worth watching before spending on retention."
    return "HIGH RISK", "var(--rose)", "#fb7185", "Likely to leave without intervention."


YESNO = ["No", "Yes"]
NO_NET = ["No", "Yes", "No internet service"]

st.markdown("""
<div class="rail">
  <div class="mk">◈</div>
  <div class="wm">ChurnGuard</div>
  <div class="rt">
    <span class="tag">Logistic Regression</span>
    <span class="tag on"><i class="blip"></i>Live scoring</span>
  </div>
</div>
""", unsafe_allow_html=True)

## ---------------------------------------------------------------------------
## annoying problem: streamlit runs the file top to bottom, but the big % at
## the top needs values from the form which is way below it. first i used
## st.empty() placeholders but that made the page jump back to the top every
## time you changed something (they go to 0 height for a split second).
## so instead every widget gets a key and we just read session_state up here.
## streamlit updates session_state before it reruns so these are current
## ---------------------------------------------------------------------------
DEFAULTS = {
    "k_contract": "Month-to-month", "k_payment": "Electronic check",
    "k_tenure": 29, "k_monthly": 70.0, "k_internet": "Fiber optic",
    "k_security": "No", "k_support": "No", "k_auto": True, "k_total": 1400.0,
    "k_backup": "No", "k_protect": "No", "k_tv": "No", "k_movies": "No",
    "k_phone": "Yes", "k_lines": "No", "k_paperless": "Yes",
    "k_gender": "Male", "k_senior": "No", "k_partner": "No", "k_depend": "No",
}


## need to set these first otherwise the widgets start on the first dropdown
## option and the sliders start at their minimum, which looks wrong
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def cur(key):
    return st.session_state.get(key, DEFAULTS[key])


tenure, monthly = cur("k_tenure"), cur("k_monthly")
total = round(monthly * tenure, 2) if cur("k_auto") else cur("k_total")

raw = {
    "gender": cur("k_gender"),
    "SeniorCitizen": 1 if cur("k_senior") == "Yes" else 0,
    "Partner": cur("k_partner"), "Dependents": cur("k_depend"), "tenure": tenure,
    "PhoneService": cur("k_phone"), "MultipleLines": cur("k_lines"),
    "InternetService": cur("k_internet"), "OnlineSecurity": cur("k_security"),
    "OnlineBackup": cur("k_backup"), "DeviceProtection": cur("k_protect"),
    "TechSupport": cur("k_support"), "StreamingTV": cur("k_tv"),
    "StreamingMovies": cur("k_movies"), "Contract": cur("k_contract"),
    "PaperlessBilling": cur("k_paperless"), "PaymentMethod": cur("k_payment"),
    "MonthlyCharges": monthly, "TotalCharges": total,
}

## validate before scoring. hard errors stop us showing a number at all, since
## a confident % from an impossible customer is worse than no number
input_errors, input_warnings = validate(raw, tenure, monthly, total)

if input_errors:
    st.error("**Cannot score this customer — please fix the following:**")
    for msg in input_errors:
        st.error(msg)
    st.caption("The score and retention levers stay hidden until the inputs are valid.")
    st.stop()

try:
    base = score(raw)
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

for msg in input_warnings:
    st.warning(msg)

pct = base * 100
band, band_var, band_hex, verdict = band_of(pct)
row = encode(derive(raw))
d = derive(raw)

## ---------------------------------------------------------------------------
## the lever thing. basically for each action the company could actually do,
## change that one field and run the model again to see how much the risk
## drops. has to happen up here because one of the cards at the top shows the
## best action
## ---------------------------------------------------------------------------
LEVERS = [
    ("Contract", "One year", "Move to a 1-year contract"),
    ("Contract", "Two year", "Move to a 2-year contract"),
    ("TechSupport", "Yes", "Add tech support"),
    ("OnlineSecurity", "Yes", "Add online security"),
    ("PaymentMethod", "Bank transfer (automatic)", "Move off electronic check"),
    ("OnlineBackup", "Yes", "Add online backup"),
    ("DeviceProtection", "Yes", "Add device protection"),
    ("PaperlessBilling", "No", "Move off paperless billing"),
]
NEEDS_NET = {"TechSupport", "OnlineSecurity", "OnlineBackup", "DeviceProtection"}

options = []
for field, value, label in LEVERS:
    if raw[field] == value:
        continue
    if field in NEEDS_NET and raw["InternetService"] == "No":
        continue
    if field == "PaymentMethod" and raw[field] != "Electronic check":
        continue
    alt = dict(raw)
    alt[field] = value
    try:
        options.append((label, score(alt)))
    except Exception:
        continue

ranked = sorted(((l, n, base - n) for l, n in options), key=lambda r: r[2], reverse=True)
shown = [r for r in ranked if r[2] > 0][:4]

## numbers for the 4 cards at the top
BASE_RATE = 0.265                       ## overall churn rate in the dataset
vs_avg = base / BASE_RATE
at_risk = monthly * base                ## rough monthly $ we could lose on this person
best_action = shown[0][0] if shown else "None available"
best_drop = f"&minus;{shown[0][2] * 100:.0f} pts" if shown else "&mdash;"

## ---------------------------------------------------------------------------
## top of the page - big headline, the % dial and the 4 stat cards
## ---------------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
  <div>
    <h1>Know who is leaving,<br>while you can <span>still act</span>.</h1>
    <p>ChurnGuard scores a customer the moment their details change, explains which
       factors drove that score, and ranks the retention actions that would move it.</p>
  </div>
  <div class="dial">
    <div class="cap">Churn probability</div>
    <div class="big" style="color:{band_var}">{pct:.0f}<i>%</i></div>
    <span class="chip" style="background:{band_hex}1f;color:{band_var}">{band}</span>
    <div class="vd">{verdict}</div>
    <div class="rail-bar"><div style="width:{max(pct,1.5):.1f}%;background:{band_var}"></div></div>
    <div class="ticks"><span>0</span><span>30</span><span>60</span><span>100</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="mets">
  <div class="met"><div class="k">Versus average customer</div>
    <div class="v">{vs_avg:.1f}&times;<em>the 26.5% base rate</em></div></div>
  <div class="met"><div class="k">Revenue at risk</div>
    <div class="v">${at_risk:,.0f}<em>per month</em></div></div>
  <div class="met"><div class="k">Best available action</div>
    <div class="v" style="font-size:1.02rem;line-height:1.3">{best_action}
      <em style="display:block;margin:.15rem 0 0;color:var(--mint)">{best_drop}</em></div></div>
  <div class="met"><div class="k">Tenure band</div>
    <div class="v" style="font-size:1.12rem">{d['tenure_group']}
      <em style="display:block;margin:.15rem 0 0">{d['num_services']} of 9 services held</em></div></div>
</div>
""", unsafe_allow_html=True)

## little floating box bottom right so you can still see the % after scrolling
st.markdown(f"""
<div class="pin">
  <div class="k"><i class="blip"></i>Churn risk</div>
  <div class="v" style="color:{band_var}">{pct:.0f}<i>%</i></div>
  <div class="t"><div style="width:{max(pct,2):.0f}%;background:{band_var}"></div></div>
  <div class="l" style="color:{band_var}">{band}</div>
</div>
""", unsafe_allow_html=True)

## ---------------------------------------------------------------------------
## the actual form. every widget has a key so the code at the top can grab the
## values next rerun - thats why nothing here gets assigned to a variable
## ---------------------------------------------------------------------------
st.markdown('<div class="sec"><h2>Customer record</h2>'
            '<p>The score updates as you edit — nothing to submit.</p></div>',
            unsafe_allow_html=True)

with st.container(border=True):
    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="grp"><b style="background:var(--blue)"></b>Account</div>',
                    unsafe_allow_html=True)
        st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="k_contract")
        st.selectbox("Payment method",
                     ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"], key="k_payment")
        st.slider("Tenure (months)", 1, 72, key="k_tenure")
        st.slider("Monthly charges ($)", 15.0, 120.0, key="k_monthly")

    with cb:
        st.markdown('<div class="grp"><b style="background:var(--violet)"></b>Internet &amp; support</div>',
                    unsafe_allow_html=True)
        st.selectbox("Internet service", ["DSL", "Fiber optic", "No"], key="k_internet")
        st.selectbox("Online security", NO_NET, key="k_security")
        st.selectbox("Tech support", NO_NET, key="k_support")
        st.checkbox("Estimate total charges from tenure", key="k_auto")
        if not st.session_state.get("k_auto", True):
            st.slider("Total charges ($)", 15.0, 9000.0, key="k_total")
        else:
            st.caption(f"Total charges: **${total:,.2f}**")

    with st.expander("Further details — defaults to the most common customer"):
        oa, ob, oc = st.columns(3)
        with oa:
            st.markdown('<div class="grp"><b style="background:var(--blue)"></b>Add-ons</div>',
                        unsafe_allow_html=True)
            st.selectbox("Online backup", NO_NET, key="k_backup")
            st.selectbox("Device protection", NO_NET, key="k_protect")
            st.selectbox("Streaming TV", NO_NET, key="k_tv")
            st.selectbox("Streaming movies", NO_NET, key="k_movies")
        with ob:
            st.markdown('<div class="grp"><b style="background:var(--violet)"></b>Phone</div>',
                        unsafe_allow_html=True)
            st.selectbox("Phone service", YESNO, key="k_phone")
            st.selectbox("Multiple lines", ["No", "Yes", "No phone service"], key="k_lines")
            st.selectbox("Paperless billing", YESNO, key="k_paperless")
        with oc:
            st.markdown('<div class="grp"><b style="background:var(--mint)"></b>Profile</div>',
                        unsafe_allow_html=True)
            st.selectbox("Gender", ["Female", "Male"], key="k_gender")
            st.selectbox("Senior citizen", YESNO, key="k_senior")
            st.selectbox("Has a partner", YESNO, key="k_partner")
            st.selectbox("Has dependents", YESNO, key="k_depend")

## ---------------------------------------------------------------------------
## works out which features pushed this particular score around. since its
## logistic regression you can just multiply the scaled value by the coefficient
## and thats the contribution. comes straight from the model, not hardcoded
## ---------------------------------------------------------------------------
FIELDS = {
    "gender": "Gender", "SeniorCitizen": "Senior citizen", "Partner": "Partner",
    "Dependents": "Dependents", "tenure_group": "Tenure group", "tenure": "Tenure",
    "PhoneService": "Phone service", "MultipleLines": "Multiple lines",
    "InternetService": "Internet", "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup", "DeviceProtection": "Device protection",
    "TechSupport": "Tech support", "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies", "Contract": "Contract",
    "PaperlessBilling": "Paperless billing", "PaymentMethod": "Payment",
    "MonthlyCharges": "Monthly charges", "TotalCharges": "Total charges",
    "num_services": "Services held",
}


def pretty(name):
    ## one-hot columns sit at 0 when the customer does NOT have that value, and a
    ## scaled 0 is negative, so those features legitimately show up as lowering
    ## risk. labelling them plainly (e.g. "Internet: Fiber optic") looks wrong next
    ## to a form that says DSL, so say "not Fiber optic" when the column is 0
    for k in sorted(FIELDS, key=len, reverse=True):
        if name == k:
            return FIELDS[k]
        if name.startswith(k + "_"):
            value = name[len(k) + 1:]
            held = bool(row[name].iloc[0]) if name in row.columns else True
            if held:
                return f"{FIELDS[k]}: {value}"
            ## "not Yes" reads badly, so flip the plain yes/no ones properly
            return f"{FIELDS[k]}: No" if value == "Yes" else f"{FIELDS[k]}: not {value}"
    return name.replace("_", " ")


def attribution(n=5):
    try:
        scaled = model.named_steps["scale"].transform(row)[0]
        coefs = model.named_steps["clf"].coef_[0]
    except Exception:
        e = pd.Series(dtype="float64")
        return e, e
    c = pd.Series(scaled * coefs, index=model.feature_names_in_)
    c = c[c.abs() > 1e-9]
    return c.nlargest(n), c.nsmallest(n).sort_values()


up, down = attribution()
peak = max(abs(up.iloc[0]) if len(up) else 0, abs(down.iloc[0]) if len(down) else 0, 1e-9)


def bars(series, colour):
    out = ""
    for nm, v in series.items():
        out += (f'<div class="dv"><span class="t">{pretty(nm)}</span>'
                f'<span class="b"><div style="width:{min(abs(v)/peak*100,100):.0f}%;'
                f'background:{colour}"></div></span></div>')
    return out


st.write("")
st.markdown('<div class="sec"><h2>Retention levers</h2>'
            '<p>Things you can actually do, ranked by how much risk each removes.</p></div>',
            unsafe_allow_html=True)
st.markdown('<div class="lead">A <strong>lever</strong> is an action the business can take, '
            'not a fact about the customer. Knowing someone is 42% likely to leave changes '
            'nothing on its own &mdash; knowing that moving them to a two-year contract cuts '
            'it to 13% is something a retention team can act on today. Each card below re-runs '
            'the model with one thing changed.</div>',
            unsafe_allow_html=True)

if shown:
    widest = shown[0][2]
    cards = "".join(
        f"""<div class="lvc{' best' if i == 0 else ''}">
              <div class="rk">{i + 1}</div>
              <div class="ac">{label}</div>
              <div class="mv"><b>{pct:.0f}%</b> &rarr; <b>{new * 100:.0f}%</b></div>
              <div class="dl">&minus;{delta * 100:.1f}<span style="font-size:.9rem"> pts</span></div>
              <div class="bar"><div style="width:{delta / widest * 100:.0f}%"></div></div>
            </div>"""
        for i, (label, new, delta) in enumerate(shown)
    )
    st.markdown(f'<div class="lv">{cards}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="note">'
        f"Each card re-runs the model on this exact customer with a single field changed, "
        f"everything else held fixed, measured against the current {pct:.0f}%. Engineered "
        f"features are recomputed per scenario, so adding a service also updates the "
        f"services-held count. These are modelled associations, not proven causal effects."
        f'</div>', unsafe_allow_html=True
    )
else:
    st.markdown('<div class="pane"><div class="empty">No modelled action lowers this '
                'customer&rsquo;s risk further — they already hold every retention lever '
                'available to them.</div></div>', unsafe_allow_html=True)

## the two red/green bar panels
st.write("")
st.markdown('<div class="sec"><h2>What drives this score</h2>'
            '<p>Each feature&rsquo;s scaled value multiplied by its fitted coefficient.</p></div>',
            unsafe_allow_html=True)

aa, ab = st.columns(2)
with aa:
    st.markdown(f'<div class="pane"><div class="ph" style="color:var(--rose)">Raising risk</div>'
                f'{bars(up, "var(--rose)")}</div>', unsafe_allow_html=True)
with ab:
    st.markdown(f'<div class="pane"><div class="ph" style="color:var(--mint)">Lowering risk</div>'
                f'{bars(down, "var(--mint)")}</div>', unsafe_allow_html=True)

with st.expander("Model detail"):
    st.caption(
        "Logistic Regression pipeline (StandardScaler + LogisticRegression), tuned with "
        "RandomizedSearchCV and trained on 7,032 telecom customers. Attribution is read "
        "from the fitted coefficients rather than fixed rules, so it reflects the model "
        "actually deployed."
    )
    st.dataframe(row.T.rename(columns={row.index[0]: "Value"}),
                 use_container_width=True, height=260)
