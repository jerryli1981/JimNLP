package edu.pengli.nlp.platform.algorithms.classify.evaluation;

import edu.pengli.nlp.platform.algorithms.classify.Clustering;


public interface ClusteringEvaluator {

	/**
	 *
	 * @param truth
	 * @param predicted
	 * @return A String summarizing the evaluation metric.
	 */
	public String evaluate (Clustering truth, Clustering predicted);
	
	public double[] getEvaluationScores (Clustering truth, Clustering predicted);
	

}

