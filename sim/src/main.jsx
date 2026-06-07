import React from "react";
import { createRoot } from "react-dom/client";
import SPHSolver from "./SPHSolver.jsx";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <SPHSolver />
  </React.StrictMode>
);
