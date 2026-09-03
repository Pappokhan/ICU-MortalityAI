(function () {
  const prob = {{PROB}};
  const tLow = {{T_LOW}};
  const tModerate = {{T_MODERATE}};
  const tHigh = {{T_HIGH}};

  const CX = 110, CY = 110, R = 90;
  const COLORS = {
    low: "#2F9E63",
    moderate: "#D9A441",
    high: "#DB6B3D",
    critical: "#C1443C",
  };

  function angleFor(v) {
    return 180 * (1 - v);
  }
  function pointAt(angleDeg) {
    const rad = (angleDeg * Math.PI) / 180;
    return [CX + R * Math.cos(rad), CY - R * Math.sin(rad)];
  }
  function zoneArc(v0, v1, color) {
    const [x1, y1] = pointAt(angleFor(v0));
    const [x2, y2] = pointAt(angleFor(v1));
    return `<path d="M ${x1} ${y1} A ${R} ${R} 0 0 1 ${x2} ${y2}"
                  fill="none" stroke="${color}" stroke-width="14" stroke-linecap="butt" />`;
  }

  const zones =
    zoneArc(0, tLow, COLORS.low) +
    zoneArc(tLow, tModerate, COLORS.moderate) +
    zoneArc(tModerate, tHigh, COLORS.high) +
    zoneArc(tHigh, 1, COLORS.critical);

  const needleAngle = angleFor(0.5);
  const targetAngle = angleFor(prob);
  const rotFor = (a) => -90 + (180 - a);

  const el = document.getElementById("gauge-root");
  el.innerHTML = `
    <svg width="220" height="128" viewBox="0 0 220 128" overflow="visible">
      ${zones}
      <line id="needle" x1="${CX}" y1="${CY}" x2="${CX}" y2="${CY - 74}"
            stroke="#ffffff" stroke-width="4" stroke-linecap="round"
            style="transform-origin:${CX}px ${CY}px; transform:rotate(${rotFor(needleAngle)}deg);" />
      <circle cx="${CX}" cy="${CY}" r="7" fill="#ffffff" />
      <circle cx="${CX}" cy="${CY}" r="3" fill="#10233B" />
      <text x="18" y="122" fill="#B9C7D6" font-size="10" font-family="IBM Plex Mono, monospace">0%</text>
      <text x="188" y="122" fill="#B9C7D6" font-size="10" font-family="IBM Plex Mono, monospace">100%</text>
    </svg>
  `;

  const needle = document.getElementById("needle");
  requestAnimationFrame(() => {
    needle.style.transition = "transform 900ms cubic-bezier(.22,.9,.3,1)";
    requestAnimationFrame(() => {
      needle.style.transform = `rotate(${rotFor(targetAngle)}deg)`;
    });
  });
})();
