package edu.pengli.nlp.platform.algorithms.ranking;

import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.util.Maths;
import edu.pengli.nlp.platform.util.matrix.Matrix;

public class PageRank {
	
	private InstanceList instances;
	private Matrix matrix;
	private double[] L;

	public PageRank() {

	}
	
	public void rank(Matrix matrix, int iterTime) {
        this.matrix = matrix;
		int N = matrix.getColumnDimension();
		L = PowerMethod(iterTime, N);

	}

	private double[] PowerMethod(int iterTime, int N) {

		Matrix p0 = new Matrix(N, 1);
		Matrix p1 = new Matrix(N, 1);

		for (int i = 0; i < N; i++) {
			p0.set(i, 0, 1 / (double) N);
		}

		Matrix mT = matrix.transpose();
		Matrix pMinus;
		p1 = mT.times(p0);
		pMinus = p1.minus(p0);
        
		int iter = 0;
		while ((iter++) < iterTime) {
			p0 = p1;
			p1 = mT.times(p0);
			//pMinus = p1.minus(p0);
		}

		return p1.getCol(0);
	}

	public double[] getScore() {
		return L;
	}
}
