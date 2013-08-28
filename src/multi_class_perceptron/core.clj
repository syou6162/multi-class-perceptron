(ns multi-class-perceptron.core
  (:refer-clojure :exclude [load-file]))

(defn parse-line
  "Function that returns [label fv]"
  [line]
  (let [[label & fv-str] (clojure.string/split line #" ")
        fv (->> fv-str
                (mapv
                 (fn [chunk]
                   (let [[k v] (->> (clojure.string/split chunk #":")
                                    (map #(Integer/parseInt %)))]
                     [k v]))))]
    [(Integer/parseInt label) fv]))

(defn load-file [filename]
  (let [content (slurp filename)]
    (->> (clojure.string/split content #"\n")
         (map parse-line)
         (vec))))

(defn inner-product [weight fv]
  (reduce
   (fn [result [k v]]
     (+ result (* (get weight k 0.0) v)))
   0.0 fv))

(defn get-label-score-pairs [weight fv]
  (reduce
   (fn [result [label w]]
     (let [score (inner-product w fv)]
       (conj result [label score])))
   []
   weight))

(defn predict-label [weight fv]
  (->> (get-label-score-pairs weight fv)
       (sort-by second >)
       (first)
       (first)))

(defn nil-safe-adder [base add] (if (nil? base) add (+ base add)))

(defn update-weight [weight gold]
  (let [gold-label (first gold)
        fv (second gold)
        y-hat (predict-label weight fv)]
    (if (= gold-label y-hat)
      weight
      (reduce
       (fn [w [k v]]
         (-> w
             (update-in [gold-label k] nil-safe-adder v)
             (update-in [y-hat k] nil-safe-adder (- v))))
       weight fv))))

(defn accuracy
  [golds predictions]
  (assert (= (count golds) (count predictions)))
  (let [n (count golds)
        sum (->> (map vector golds predictions)
                 (map (fn [[g p]] (if (= g p) 1.0 0.0)))
                 (reduce + 0.0))]
    (/ sum n)))

(defn -main [& args]
  (let [training-data (load-file "resources/news20")
        test-data (load-file "resources/news20.t")]
    (loop [iter 0
           weight {}]
      (let [new-weight (reduce
                        (fn [result gold]
                          (update-weight result gold))
                        weight training-data)
            predictions (->> test-data
                             (map second)
                             (mapv (partial predict-label new-weight)))]
        (println (str iter ": " (accuracy (mapv first test-data)
                                          predictions)))
        (recur (inc iter) new-weight)))))
