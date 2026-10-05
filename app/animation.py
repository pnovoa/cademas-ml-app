import streamlit.components.v1 as components


def render_animated_header():
    """
    Render the CADEMAS-ML pipeline animation:
    inputs → Inputs and Parameter Settings → Prioritization engine → analysis views.
    """
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@100..900&display=swap');
        </style>
        <style>
            body {
                margin: 0;
                padding: 0;
                background-color: transparent;
                font-family: 'Geist Mono', monospace;
                overflow: hidden;
            }
            .container {
                width: 100%;
                height: 430px;
                background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 55%, #f8fafc 100%);
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            svg { width: 100%; height: 100%; max-width: 820px; }

            .node-rect {
                fill: #ffffff;
                stroke: #cbd5e1;
                stroke-width: 2px;
                rx: 6px;
            }
            .text-title {
                fill: #111827;
                font-size: 11px;
                font-weight: 600;
                font-family: 'Geist Mono', monospace;
                pointer-events: none;
            }
            .text-sub {
                fill: #64748b;
                font-size: 9px;
                font-family: 'Geist Mono', monospace;
                pointer-events: none;
            }
            .stroke-input { stroke: #94a3b8; }
            .stroke-process { stroke: #475569; }
            .stroke-hybrid { stroke: #7c3aed; }
            .stroke-output { stroke: #059669; }

            .path-line {
                fill: none;
                stroke: #94a3b8;
                stroke-width: 2px;
                opacity: 0.6;
            }
            .dot { filter: drop-shadow(0 0 4px rgba(15,23,42,0.25)); }
            .dot-input { fill: #64748b; }
            .dot-process { fill: #475569; }
            .dot-hybrid { fill: #7c3aed; }
            .dot-output { fill: #059669; }
        </style>
    </head>
    <body>
        <div class="container">
            <svg viewBox="0 0 530 410" preserveAspectRatio="xMidYMid meet">
                <defs>
                    <!-- Inputs → Parameter Settings -->
                    <path id="in1" d="M 70 62 L 70 100 L 265 100 L 265 118" />
                    <path id="in2" d="M 200 62 L 200 100 L 265 100 L 265 118" />
                    <path id="in3" d="M 330 62 L 330 100 L 265 100 L 265 118" />
                    <path id="in4" d="M 460 62 L 460 100 L 265 100 L 265 118" />
                    <!-- Parameter Settings → Prioritization engine -->
                    <path id="settings-engine" d="M 265 178 L 265 208" />
                    <!-- Prioritization engine → Outputs -->
                    <path id="eng-out1" d="M 220 278 L 220 300 L 57 300 L 57 322" />
                    <path id="eng-out2" d="M 240 278 L 240 300 L 161 300 L 161 322" />
                    <path id="eng-out3" d="M 265 278 L 265 300 L 265 322" />
                    <path id="eng-out4" d="M 290 278 L 290 300 L 369 300 L 369 322" />
                    <path id="eng-out5" d="M 310 278 L 310 300 L 473 300 L 473 322" />
                </defs>

                <!-- Connector lines -->
                <path d="M 70 62 L 70 100 L 265 100 L 265 118" class="path-line" />
                <path d="M 200 62 L 200 100 L 265 100 L 265 118" class="path-line" />
                <path d="M 330 62 L 330 100 L 265 100 L 265 118" class="path-line" />
                <path d="M 460 62 L 460 100 L 265 100 L 265 118" class="path-line" />
                <path d="M 265 178 L 265 208" class="path-line" />
                <path d="M 220 278 L 220 300 L 57 300 L 57 322" class="path-line" />
                <path d="M 240 278 L 240 300 L 161 300 L 161 322" class="path-line" />
                <path d="M 265 278 L 265 300 L 265 322" class="path-line" />
                <path d="M 290 278 L 290 300 L 369 300 L 369 322" class="path-line" />
                <path d="M 310 278 L 310 300 L 473 300 L 473 322" class="path-line" />

                <!-- Level 1: four inputs -->
                <g transform="translate(15, 20)">
                    <rect width="110" height="42" class="node-rect stroke-input" />
                    <text x="55" y="14" text-anchor="middle" class="text-title">Model</text>
                    <text x="55" y="27" text-anchor="middle" class="text-title">config</text>
                    <text x="55" y="38" text-anchor="middle" class="text-sub">JSON</text>
                </g>
                <g transform="translate(145, 20)">
                    <rect width="110" height="42" class="node-rect stroke-input" />
                    <text x="55" y="14" text-anchor="middle" class="text-title">Context</text>
                    <text x="55" y="27" text-anchor="middle" class="text-title">config</text>
                    <text x="55" y="38" text-anchor="middle" class="text-sub">JSON</text>
                </g>
                <g transform="translate(275, 20)">
                    <rect width="110" height="42" class="node-rect stroke-input" />
                    <text x="55" y="14" text-anchor="middle" class="text-title">MOJO</text>
                    <text x="55" y="27" text-anchor="middle" class="text-title">models</text>
                    <text x="55" y="38" text-anchor="middle" class="text-sub">.zip</text>
                </g>
                <g transform="translate(405, 20)">
                    <rect width="110" height="42" class="node-rect stroke-input" />
                    <text x="55" y="14" text-anchor="middle" class="text-title">Case</text>
                    <text x="55" y="27" text-anchor="middle" class="text-title">dataset</text>
                    <text x="55" y="38" text-anchor="middle" class="text-sub">CSV</text>
                </g>

                <!-- Level 2: Inputs and Parameter Settings -->
                <g transform="translate(125, 118)">
                    <rect width="280" height="60" class="node-rect stroke-process" />
                    <text x="140" y="18" text-anchor="middle" class="text-title">Inputs and Parameter Settings</text>
                    <text x="140" y="36" text-anchor="middle" class="text-sub">Validation and Preprocessing</text>
                    <text x="140" y="50" text-anchor="middle" class="text-sub">sidebar configuration</text>
                </g>

                <!-- Level 3: Prioritization engine -->
                <g transform="translate(125, 208)">
                    <rect width="280" height="70" class="node-rect stroke-hybrid" />
                    <text x="140" y="20" text-anchor="middle" class="text-title">Prioritization engine</text>
                    <text x="140" y="38" text-anchor="middle" class="text-sub">Decision integration,</text>
                    <text x="140" y="52" text-anchor="middle" class="text-sub">Context modulator</text>
                </g>

                <!-- Level 4: analysis views -->
                <g transform="translate(10, 322)">
                    <rect width="94" height="52" class="node-rect stroke-output" />
                    <text x="47" y="18" text-anchor="middle" class="text-title">Overview</text>
                    <text x="47" y="34" text-anchor="middle" class="text-sub">priority</text>
                    <text x="47" y="46" text-anchor="middle" class="text-sub">scores</text>
                </g>
                <g transform="translate(114, 322)">
                    <rect width="94" height="52" class="node-rect stroke-output" />
                    <text x="47" y="18" text-anchor="middle" class="text-title">Models</text>
                    <text x="47" y="34" text-anchor="middle" class="text-sub">weights &amp;</text>
                    <text x="47" y="46" text-anchor="middle" class="text-sub">predictions</text>
                </g>
                <g transform="translate(218, 322)">
                    <rect width="94" height="52" class="node-rect stroke-output" />
                    <text x="47" y="18" text-anchor="middle" class="text-title">Context</text>
                    <text x="47" y="34" text-anchor="middle" class="text-sub">rules &amp;</text>
                    <text x="47" y="46" text-anchor="middle" class="text-sub">alignment</text>
                </g>
                <g transform="translate(322, 322)">
                    <rect width="94" height="52" class="node-rect stroke-output" />
                    <text x="47" y="18" text-anchor="middle" class="text-title">Explain</text>
                    <text x="47" y="34" text-anchor="middle" class="text-sub">case-level</text>
                    <text x="47" y="46" text-anchor="middle" class="text-sub">XAI</text>
                </g>
                <g transform="translate(426, 322)">
                    <rect width="94" height="52" class="node-rect stroke-output" />
                    <text x="47" y="18" text-anchor="middle" class="text-title">Robustness</text>
                    <text x="47" y="34" text-anchor="middle" class="text-sub">Q_mod</text>
                    <text x="47" y="46" text-anchor="middle" class="text-sub">stability</text>
                </g>

                <!-- Animated particles: inputs → settings -->
                <circle r="4" class="dot dot-input">
                    <animateMotion dur="8s" repeatCount="indefinite" calcMode="linear"
                                   keyTimes="0;0.22;1" keyPoints="0;1;1">
                        <mpath href="#in1"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="1;1;1;0;0" dur="8s" repeatCount="indefinite"
                             keyTimes="0;0.02;0.20;0.22;1" />
                </circle>
                <circle r="4" class="dot dot-input">
                    <animateMotion dur="8s" repeatCount="indefinite" calcMode="linear"
                                   keyTimes="0;0.22;1" keyPoints="0;1;1">
                        <mpath href="#in2"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="1;1;1;0;0" dur="8s" repeatCount="indefinite"
                             keyTimes="0;0.02;0.20;0.22;1" />
                </circle>
                <circle r="4" class="dot dot-input">
                    <animateMotion dur="8s" repeatCount="indefinite" calcMode="linear"
                                   keyTimes="0;0.22;1" keyPoints="0;1;1">
                        <mpath href="#in3"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="1;1;1;0;0" dur="8s" repeatCount="indefinite"
                             keyTimes="0;0.02;0.20;0.22;1" />
                </circle>
                <circle r="4" class="dot dot-input">
                    <animateMotion dur="8s" repeatCount="indefinite" calcMode="linear"
                                   keyTimes="0;0.22;1" keyPoints="0;1;1">
                        <mpath href="#in4"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="1;1;1;0;0" dur="8s" repeatCount="indefinite"
                             keyTimes="0;0.02;0.20;0.22;1" />
                </circle>

                <!-- Settings → Prioritization engine -->
                <circle r="5" class="dot dot-process">
                    <animateMotion dur="8s" repeatCount="indefinite" calcMode="linear"
                                   keyTimes="0;0.28;0.42;1" keyPoints="0;0;1;1">
                        <mpath href="#settings-engine"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;0;1;1;0;0" dur="8s" repeatCount="indefinite"
                             keyTimes="0;0.28;0.30;0.40;0.42;1" />
                </circle>

                <!-- Prioritization engine → outputs -->
                <circle r="4" class="dot dot-hybrid">
                    <animateMotion dur="8s" repeatCount="indefinite" calcMode="linear"
                                   keyTimes="0;0.52;0.78;1" keyPoints="0;0;1;1">
                        <mpath href="#eng-out1"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;0;1;1;0;0" dur="8s" repeatCount="indefinite"
                             keyTimes="0;0.52;0.54;0.76;0.78;1" />
                </circle>
                <circle r="4" class="dot dot-output">
                    <animateMotion dur="8s" repeatCount="indefinite" calcMode="linear"
                                   keyTimes="0;0.54;0.80;1" keyPoints="0;0;1;1">
                        <mpath href="#eng-out2"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;0;1;1;0;0" dur="8s" repeatCount="indefinite"
                             keyTimes="0;0.54;0.56;0.78;0.80;1" />
                </circle>
                <circle r="4" class="dot dot-output">
                    <animateMotion dur="8s" repeatCount="indefinite" calcMode="linear"
                                   keyTimes="0;0.56;0.82;1" keyPoints="0;0;1;1">
                        <mpath href="#eng-out3"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;0;1;1;0;0" dur="8s" repeatCount="indefinite"
                             keyTimes="0;0.56;0.58;0.80;0.82;1" />
                </circle>
                <circle r="4" class="dot dot-output">
                    <animateMotion dur="8s" repeatCount="indefinite" calcMode="linear"
                                   keyTimes="0;0.58;0.84;1" keyPoints="0;0;1;1">
                        <mpath href="#eng-out4"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;0;1;1;0;0" dur="8s" repeatCount="indefinite"
                             keyTimes="0;0.58;0.60;0.82;0.84;1" />
                </circle>
                <circle r="4" class="dot dot-output">
                    <animateMotion dur="8s" repeatCount="indefinite" calcMode="linear"
                                   keyTimes="0;0.60;0.86;1" keyPoints="0;0;1;1">
                        <mpath href="#eng-out5"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;0;1;1;0;0" dur="8s" repeatCount="indefinite"
                             keyTimes="0;0.60;0.62;0.84;0.86;1" />
                </circle>
            </svg>
        </div>
    </body>
    </html>
    """
    components.html(html_code, height=440, scrolling=False)
