package edu.pengli.nlp.platform.types;

/**
	 Computes
	 1 - [<x,y> / sqrt (<x,x>*<y,y>)]
	 aka 1 - cosine similarity
 */

public class NormalizedDotProductMetric implements Metric {

	public NormalizedDotProductMetric () {

	}
	
	public double distance (SparseVector a, SparseVector b) {
	    //		double ret = a.dotProduct (b) /
	    //								 Math.sqrt (a.dotProduct (a) * b.dotProduct (b));
	    // gmann : twoNorm() more efficient than a.dotProduct(a)
	    double ret = a.dotProduct(b) / (a.twoNorm()*b.twoNorm());
	    return 1.0 - ret;
	}
	
}

