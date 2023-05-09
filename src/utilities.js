import * as tf from "@tensorflow/tfjs";


export const drawEyes = (face, ctx) => {
  const annots = face.annotations

  const keypoints = [...annots.leftEyeIris, ...annots.leftEyeLower0, ...annots.leftEyeLower1, ...annots.leftEyebrowUpper,
  ...annots.rightEyeIris, ...annots.rightEyeLower0, ...annots.rightEyeLower1, ...annots.rightEyebrowUpper]

  for (let i = 0; i < keypoints.length; i++) {
    const x = keypoints[i][0];
    const y = keypoints[i][1];
    ctx.beginPath();
    ctx.arc(x, y, 1 /* radius */, 0, 3 * Math.PI);
    ctx.fillStyle = "aqua";
    ctx.fill();
  }
}

export const drawGazePoint = (x, y, ctx) => {
  const x_final = Math.max(Math.min(window.innerWidth - 20, x), 20);
  const y_final = Math.max(Math.min(window.innerHeight - 20, y), 20);
  //console.log(x);
  //console.log(y);
  ctx.beginPath();
  ctx.arc(x_final, y_final, 10 /* radius */, 0, 3 * Math.PI);
  ctx.fillStyle = "#FF0000";
  ctx.fill();
}