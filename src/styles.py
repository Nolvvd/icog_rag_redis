"""Visual identity and styling for CircuitMind — Engineering Design Intelligence.
Implements the refined dark navy theme with repeating '+' crosshair blueprint grid,
high-readability typography, and professional chat bubble styling.
"""

CIRCUITMIND_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400;1,600&display=swap');

/* Main Canvas: Dark Navy Canvas with Subtle '+' Crosshair Blueprint Grid */
.stApp {
    background-color: #080C14 !important;
    background-image: url("data:image/svg+xml,%3Csvg width='36' height='36' viewBox='0 0 36 36' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M17 14h2v8h-2v-8zm-3 3h8v2h-8v-2z' fill='%2338BDF8' fill-opacity='0.16'/%3E%3C/svg%3E") !important;
    background-size: 36px 36px !important;
    background-position: 0 0 !important;
    color: #F8FAFC !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

/* Full Width Workspace with Minimal Edge Margins */
.main .block-container {
    max-width: 100% !important;
    padding-top: 1.2rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2.2rem !important;
    padding-right: 2.2rem !important;
}

/* Sidebar: Slightly lighter, muted dark navy with subtle right border */
section[data-testid="stSidebar"] {
    background-color: #111827 !important;
    border-right: 1px solid #1F2937 !important;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.4rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 1.3rem !important;
    padding-right: 1.3rem !important;
}

/* Sidebar Typography */
.sidebar-heading {
    font-size: 20px !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
    letter-spacing: -0.015em !important;
    margin-top: 0.8rem !important;
    margin-bottom: 0.4rem !important;
}

.sidebar-desc {
    font-size: 13.5px !important;
    color: #94A3B8 !important;
    line-height: 1.5 !important;
    margin-bottom: 0.6rem !important;
}

/* Sidebar Document Info Card */
.sidebar-corpus-box {
    background-color: #0F172A;
    border: 1px solid #1F2937;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-top: 0.5rem;
    margin-bottom: 0.75rem;
}

.sidebar-corpus-name {
    font-size: 15px;
    font-weight: 600;
    color: #F8FAFC;
    margin-top: 4px;
    word-break: break-all;
}

.sidebar-corpus-meta {
    font-size: 13px;
    color: #94A3B8;
    font-family: 'Fira Code', monospace;
    margin-top: 3px;
}

/* Sidebar Dropzone */
.sidebar-dropzone {
    border: 1.5px dashed #38BDF8;
    background: rgba(15, 23, 42, 0.7);
    border-radius: 8px;
    padding: 1rem 0.8rem;
    text-align: center;
    margin-top: 0.4rem;
    margin-bottom: 0.75rem;
    transition: all 0.2s ease;
}

.sidebar-dropzone:hover {
    border-color: #34D399;
    background: rgba(16, 185, 129, 0.08);
}

/* Main Area Branding Typography */
.brand-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(56, 189, 248, 0.35);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11px;
    font-weight: 700;
    color: #38BDF8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.6rem;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.15);
}

.main-title {
    font-size: 48px !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #FFFFFF 40%, #93C5FD 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    letter-spacing: -0.03em !important;
    line-height: 1.15 !important;
    margin-top: 0 !important;
    margin-bottom: 0.35rem !important;
}

.main-subtitle {
    font-size: 18.5px !important;
    color: #94A3B8 !important;
    font-weight: 400 !important;
    letter-spacing: 0.01em !important;
    margin-bottom: 1.5rem !important;
}

/* System Status Ribbon */
.system-status-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 14px;
    background-color: #111827;
    border: 1px solid #1F2937;
    border-radius: 8px;
    padding: 0.7rem 1.2rem;
    margin-bottom: 1.6rem;
    font-size: 14px;
}

.status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-weight: 500;
    color: #CBD5E1;
}

@keyframes pulse-green {
    0% {
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    }
    70% {
        box-shadow: 0 0 0 6px rgba(16, 185, 129, 0);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
    }
}

.status-dot-green {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #10B981;
    display: inline-block;
    animation: pulse-green 2.2s infinite;
}

.status-dot-amber {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #F59E0B;
    display: inline-block;
}

.status-dot-red {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #EF4444;
    display: inline-block;
}

/* Headings */
h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
    letter-spacing: -0.02em !important;
}

h2 {
    font-size: 24px !important;
    margin-top: 1.4rem !important;
    margin-bottom: 0.5rem !important;
}

h3 {
    font-size: 20px !important;
    margin-top: 1.1rem !important;
    margin-bottom: 0.5rem !important;
}

p, li {
    font-size: 18px !important;
    line-height: 1.74 !important;
    color: #E2E8F0 !important;
}

/* User Message Box (Right-aligned chatbot bubble) */
.user-chat-bubble {
    background-color: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 14px 20px !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
    margin-top: 1rem !important;
    margin-bottom: 1.2rem !important;
    text-align: left !important;
}

.chat-sender-label {
    font-size: 11.5px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    margin-bottom: 5px !important;
}

.user-label {
    color: #38BDF8 !important;
}

.assistant-label {
    color: #34D399 !important;
}

.chat-bubble-text {
    font-size: 17.5px !important;
    font-weight: 500 !important;
    line-height: 1.55 !important;
    color: #F8FAFC !important;
}

/* Assistant Container Box (Left-aligned chatbot card) */
[class*="st-key-assistant_answer_box"],
div[data-testid="stVerticalBlockBorderWrapper"]:has(.assistant-header-pill),
div[data-testid="stVerticalBlock"]:has(.assistant-header-pill) {
    background-color: #111827 !important;
    border: 1px solid #1F2937 !important;
    border-left: 3.5px solid #38BDF8 !important;
    border-radius: 16px 16px 16px 4px !important;
    padding: 1.6rem 1.8rem !important;
    margin-bottom: 1.5rem !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.45) !important;
}

[class*="st-key-assistant_answer_box"] > div {
    background-color: #111827 !important;
}

.assistant-header-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-size: 12px;
    font-weight: 700;
    color: #38BDF8;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-family: 'Fira Code', monospace;
    margin-bottom: 14px;
}

/* Obsidian Code Blocks & Callouts */
code {
    font-family: 'Fira Code', monospace !important;
    background: #0B0F17 !important;
    color: #38BDF8 !important;
    border: 1px solid #1E293B !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
    font-size: 0.9em !important;
}

pre {
    background: #0B0F17 !important;
    border: 1px solid #1E293B !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4) !important;
}

pre code {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    color: #E2E8F0 !important;
}

blockquote {
    background: rgba(15, 23, 42, 0.6) !important;
    border-left: 3.5px solid #10B981 !important;
    border-radius: 0 8px 8px 0 !important;
    margin: 1rem 0 !important;
    padding: 0.75rem 1.1rem !important;
    color: #CBD5E1 !important;
}

/* Badges */
.badge-hit {
    display: inline-flex;
    align-items: center;
    background: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.35);
    border-radius: 5px;
    padding: 3px 9px;
    font-size: 13.5px;
    font-weight: 700;
}

.badge-miss {
    display: inline-flex;
    align-items: center;
    background: rgba(14, 165, 233, 0.15);
    color: #38BDF8;
    border: 1px solid rgba(14, 165, 233, 0.35);
    border-radius: 5px;
    padding: 3px 9px;
    font-size: 13.5px;
    font-weight: 700;
}

.badge-neutral {
    display: inline-flex;
    align-items: center;
    background: rgba(100, 116, 139, 0.2);
    color: #CBD5E1;
    border: 1px solid rgba(100, 116, 139, 0.4);
    border-radius: 5px;
    padding: 3px 9px;
    font-size: 13px;
    font-weight: 600;
}

/* Markdown Tables in Answer */
table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 1.2rem 0 !important;
    font-size: 16px !important;
    background: #0F172A !important;
    border: 1px solid #1F2937 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

table th {
    background-color: #1E293B !important;
    color: #38BDF8 !important;
    font-weight: 700 !important;
    text-align: left !important;
    padding: 0.85rem 1.05rem !important;
    border-bottom: 2px solid #334155 !important;
    border-right: 1px solid #1F2937 !important;
    font-size: 15.5px !important;
}

table td {
    padding: 0.85rem 1.05rem !important;
    border-bottom: 1px solid #1F2937 !important;
    border-right: 1px solid #1F2937 !important;
    color: #E2E8F0 !important;
    line-height: 1.62 !important;
    font-size: 15.5px !important;
}

table tr:last-child td {
    border-bottom: none !important;
}

table tr:nth-child(even) td {
    background-color: #0B1120 !important;
}

table tr:hover td {
    background-color: #1E293B !important;
}

/* Text Input Field */
.stTextInput input {
    background-color: #111827 !important;
    border: 1.5px solid #1F2937 !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
    font-size: 18px !important;
    padding: 13px 18px !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3) !important;
}

.stTextInput input:focus {
    border-color: #38BDF8 !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.25) !important;
}

/* Primary Action Button (Ask CircuitMind →) */
button[kind="primary"] {
    background: linear-gradient(135deg, #EF4444 0%, #F43F5E 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    padding: 0.7rem 1.5rem !important;
    box-shadow: 0 2px 10px rgba(239, 68, 68, 0.35) !important;
    transition: all 0.15s ease !important;
}

button[kind="primary"]:hover {
    background: linear-gradient(135deg, #DC2626 0%, #E11D48 100%) !important;
    box-shadow: 0 4px 14px rgba(239, 68, 68, 0.5) !important;
}

/* Secondary & Preset Buttons */
button[kind="secondary"] {
    background-color: #111827 !important;
    border: 1px solid #1F2937 !important;
    color: #E2E8F0 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14.5px !important;
    padding: 0.65rem 1.15rem !important;
    transition: all 0.2s ease !important;
}

button[kind="secondary"]:hover {
    background-color: #1E293B !important;
    border-color: #38BDF8 !important;
    color: #38BDF8 !important;
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.2) !important;
}

/* Telemetry Card in Expander */
.telemetry-card {
    background-color: #0F172A;
    border: 1px solid #1F2937;
    border-radius: 8px;
    padding: 0.95rem 1.1rem;
    margin-top: 0.6rem;
    margin-bottom: 0.8rem;
}

/* Streamlit Expander Overrides for Dark Navy */
.streamlit-expanderHeader {
    background-color: #0F172A !important;
    border: 1px solid #1F2937 !important;
    border-radius: 8px !important;
    color: #F8FAFC !important;
    font-size: 15.5px !important;
    font-weight: 600 !important;
}

.streamlit-expanderContent {
    background-color: #0B0F19 !important;
    border: 1px solid #1F2937 !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
    padding: 1rem 1.2rem !important;
}
</style>
"""
