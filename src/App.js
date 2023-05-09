import React, { useEffect, useRef, useState } from "react";
import './App.css'
import * as tf from "@tensorflow/tfjs";
import * as facemesh from "@tensorflow-models/face-landmarks-detection";
import { drawEyes, drawGazePoint } from "./utilities";
import { GazeNet } from "./GazeNet";

let model = null
let gazenet = null

const increaseClickCount = (arr, i) => {
  const newArr = [...arr]
  newArr[i] += 1
  return newArr
}

function App() {

  const [clickCount, setClickCount] = useState(Array(9).fill(0))


  //const gazeNet = useRef(new GazeNet())

  const videoRef = useRef(null);
  const photoRef = useRef(null);
  const backcanvasRef = useRef(null)

  useEffect(() => {
    modelSetup()
  }, [])

  useEffect(() => {
    getVideo();
  }, [videoRef]);
  useEffect(() => {
    backSetup();
  }, [backcanvasRef])

  const modelSetup = async () => {
    model = await facemesh.load(facemesh.SupportedPackages.mediapipeFacemesh);
    gazenet = new GazeNet();
  }

  const backSetup = async () => {

    const background = backcanvasRef.current;
    var ctx = background.getContext("2d");
    ctx.canvas.width = window.innerWidth;
    ctx.canvas.height = window.innerHeight;

  }

  const getVideo = () => {
    navigator.mediaDevices
      .getUserMedia({ video: { width: 200 } })
      .then(stream => {
        let video = videoRef.current;
        video.srcObject = stream;
        video.play();
      })
      .catch(err => {
        console.error("error:", err);
      });
  };

  const paintToCanvas = async () => {
    let video = videoRef.current;
    let photo = photoRef.current;
    let background = backcanvasRef.current;
    let backCtx = background.getContext("2d");
    let ctx = photo.getContext("2d");

    const width = video.videoWidth;
    const height = video.videoHeight;
    photo.width = width;
    photo.height = height;

    return setInterval(async () => {
      //draw the point on the screen
      backCtx.clearRect(0, 0, background.width, background.height);
      //draw the facial landmarks
      ctx.drawImage(video, 0, 0, width, height);
      let face = undefined;
      if (model != null) {
        face = await model.estimateFaces({ input: video });
      }


      if (typeof face !== "undefined" && typeof face[0] !== "undefined") {
        if (gazenet != null) {
          const gaze = await gazenet.predict(face[0]);

          drawGazePoint(gaze[0], gaze[1], backCtx);
        }
        requestAnimationFrame(() => { drawEyes(face[0], ctx) });
      }
    }, 50);
  };
  const clickHandler = async (index, position) => {
    setClickCount(increaseClickCount(clickCount, index));
    const face = await model.estimateFaces({ input: videoRef.current });
    if (typeof face !== "undefined" && typeof face[0] !== "undefined") {
      gazenet.train(face[0], position);
    }

  }

  return (

    <div>

      <video
        onCanPlay={() => paintToCanvas()}
        ref={videoRef}
        className="player"
      />
      <canvas ref={photoRef}></canvas>
      <canvas ref={backcanvasRef}></canvas>
      <button className="point topButton" onClick={() => { clickHandler(0, [window.innerWidth / 2, 40]) }}>{clickCount[0]}</button>
      <button className="point rightButton" onClick={() => { clickHandler(1, [window.innerWidth - 40, window.innerHeight / 2]) }}>{clickCount[1]}</button>
      <button className="point botButton" onClick={() => { clickHandler(2, [window.innerWidth / 2, window.innerHeight - 10]) }}>{clickCount[2]}</button>
      <button className="point leftButton" onClick={() => { clickHandler(3, [0 + 40, window.innerHeight / 2]) }}>{clickCount[3]}</button>
      <button className="point topLeftButton" onClick={() => { clickHandler(4, [40, 40]) }}>{clickCount[4]}</button>
      <button className="point topRightButton" onClick={() => { clickHandler(5, [window.innerWidth - 40, 40]) }}>{clickCount[5]}</button>
      <button className="point botRightButton" onClick={() => { clickHandler(6, [window.innerWidth - 40, window.innerHeight - 40]) }}>{clickCount[6]}</button>
      <button className="point botLeftButton" onClick={() => { clickHandler(7, [40, window.innerHeight - 40]) }}>{clickCount[7]}</button>
      <button className="point centerButton" onClick={() => { clickHandler(8, [window.innerWidth / 2, window.innerHeight / 2]) }}>{clickCount[8]}</button>

    </div>
  );
};

export default App;