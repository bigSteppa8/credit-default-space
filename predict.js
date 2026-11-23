/*
 * Logistic regression inference for the credit card default model.
 *
 * Re-implements the scikit-learn Pipeline exactly:
 *   1. StandardScaler on the 14 continuous features: z = (x - mean) / scale
 *   2. OneHotEncoder(handle_unknown="ignore") on the 9 coded features:
 *      a known category adds its coefficient, an unknown one adds nothing
 *   3. LogisticRegression: p = 1 / (1 + exp(-(intercept + sum of terms)))
 *
 * Works in the browser (window.CreditModel) and in Node (module.exports),
 * so the CI pipeline can test the same file the Space serves.
 */
(function (root) {
  "use strict";

  function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
  }

  function predict(params, record) {
    let logit = params.intercept;
    const contributions = {};

    for (const [name, p] of Object.entries(params.numeric)) {
      const raw = Number(record[name]);
      if (!Number.isFinite(raw)) {
        throw new Error("Missing or non-numeric value for " + name);
      }
      const term = ((raw - p.mean) / p.scale) * p.coef;
      contributions[name] = term;
      logit += term;
    }

    for (const [name, weights] of Object.entries(params.categorical)) {
      const key = String(Math.trunc(Number(record[name])));
      const term = Object.prototype.hasOwnProperty.call(weights, key) ? weights[key] : 0;
      // shown relative to a typical training client; the logit uses the raw weight
      const base = params.categorical_baseline ? params.categorical_baseline[name] : 0;
      contributions[name] = term - base;
      logit += term;
    }

    const probability = sigmoid(logit);
    return {
      probability: probability,
      prediction: probability >= params.threshold ? 1 : 0,
      logit: logit,
      contributions: contributions,
    };
  }

  const api = { predict: predict, sigmoid: sigmoid };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    root.CreditModel = api;
  }
})(typeof self !== "undefined" ? self : this);
