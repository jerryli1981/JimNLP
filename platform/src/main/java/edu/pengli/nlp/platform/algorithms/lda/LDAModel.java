package edu.pengli.nlp.platform.algorithms.lda;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Random;
import java.util.Set;
import java.util.Vector;

import edu.pengli.nlp.platform.types.Alphabet;
import edu.pengli.nlp.platform.types.FeatureSequence;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.util.RankMap;

public class LDAModel implements Serializable {

	private InstanceList instances; // here means collection of documents

	private int K; // number of topics
	private int D; // number of documents
	private int V; // vocabulary size
	private double alpha;
	private double beta;
	private int numIters; // number of Gibbs sampling iteration

	// Estimated/Inferenced parameters
	private double[][] theta; // theta: document-topic distributions, size M x K
	private double[][] phi; // phi: topic-word distributions, size K x V

	private int[][] z; // topic assignment for word of document z_{d,n}

	private int[][] nw; // nw[i][j]: number of instances of word/term i
						// assigned to topic j, size V x K
	private int[][] nd; // nd[i][j]: number of words in document i assigned to
						// topic j, size M x K
	private int[] nwsum; // nwsum[j]: total number of words assigned to topic
							// j, size K
	private int[] ndsum; // ndsum[i]: total number of words in document i,
							// size M

	// temp variables for sampling
	private double[] p; // p(z|w)

	public LDAModel(int numTopics, double alpha, double beta, int numIters) {
		this.K = numTopics;
		this.alpha = alpha;
		this.beta = beta;
		this.numIters = numIters;
	}

	public InstanceList getInstanceList() {
		return instances;
	}

	public int[][] getTopicWordsAssignment(){
		return z;
	}
	public Alphabet getAlphabet() {
		return instances.getDataAlphabet();
	}

	public double[][] getTopicWordDistribution() {
		return phi;
	}

	public void initEstimate(InstanceList instances) {
		this.instances = instances;
		Alphabet dict = instances.getDataAlphabet();

		V = dict.size();
		D = instances.size();

		int m, n, w, k;
		p = new double[K];

		nw = new int[V][K]; // number of word i assign to topic j
		for (w = 0; w < V; w++) {
			for (k = 0; k < K; k++) {
				nw[w][k] = 0;
			}
		}

		nd = new int[D][K]; // number of words in document i assign to topic j
		for (m = 0; m < D; m++) {
			for (k = 0; k < K; k++) {
				nd[m][k] = 0;
			}
		}

		nwsum = new int[K]; // total number of words assign to topic j
		for (k = 0; k < K; k++) {
			nwsum[k] = 0;
		}

		ndsum = new int[D]; // total number of words in document i
		for (m = 0; m < D; m++) {
			ndsum[m] = 0;
		}

		z = new int[D][]; // topic assignments for words
		for (m = 0; m < D; m++) {
			FeatureSequence fs = (FeatureSequence) instances.get(m).getData();
			int N = fs.size();
			z[m] = new int[N];

			// initilize for z
			for (n = 0; n < N; n++) {
				int topic = (int) Math.floor(Math.random() * K);
				z[m][n] = topic;

				// number of word i assign to topic j
				nw[fs.getIndexAtPosition(n)][topic] += 1;
				// number of words in document i assigned to topic j
				nd[m][topic] += 1;
				// total number of words assigned to topic j
				nwsum[topic] += 1;
			}
			// total number of words in document i
			ndsum[m] = N;
		}

		theta = new double[D][K];
		phi = new double[K][V];

	}

	public void estimate() {

		for (int i = 0; i < numIters; i++) {
		//	System.out.println("Iteration " + (i + 1) + " ...");
			sweep();
		}// end iterations

	//	System.out.println("Taking samples...");
		for (int i = 0; i < 100; i++) {
			sweep();
			if (i % 10 == 0) {
				computeTheta();
				computePhi();
			}
		}
		computeTheta();
		computePhi();

	}

	private void sweep() {
		// for all z_i
		for (int m = 0; m < D; m++) {
			for (int n = 0; n < ndsum[m]; n++) {
				// z_i = z[m][n]
				// sample from p(z_i|z_-i, w)
				int topic = sampling_z(m, n);
				z[m][n] = topic;
			}// end for each word
		}// end for each document
	}

	private void computeTheta() {
		for (int m = 0; m < D; m++) {
			for (int k = 0; k < K; k++) {
				theta[m][k] = (nd[m][k] + alpha) / (ndsum[m] + K * alpha);
			}
		}
	}

	private void computePhi() {
		for (int k = 0; k < K; k++) {
			for (int w = 0; w < V; w++) {
				phi[k][w] = (nw[w][k] + beta) / (nwsum[k] + V * beta);
			}
		}
	}

	private int sampling_z(int m, int n) {

		// remove z_i from the count variable
		int topic = z[m][n];
		FeatureSequence fs = (FeatureSequence) instances.get(m).getData();
		int w = fs.getIndexAtPosition(n);

		// below means not include the current topic assignment for word
		nw[w][topic] -= 1;
		nd[m][topic] -= 1;
		nwsum[topic] -= 1;
		ndsum[m] -= 1;

		double Vbeta = V * beta;
		double Kalpha = K * alpha;

		// do multinominal sampling via cumulative method
		for (int k = 0; k < K; k++) {
			p[k] = (nw[w][k] + beta) / (nwsum[k] + Vbeta) * (nd[m][k] + alpha)
					/ (ndsum[m] + Kalpha);
		}

		// cumulate multinomial parameters //why need to do this?
		for (int k = 1; k < K; k++) {
			p[k] += p[k - 1];
		}

		// scaled sample because of unnormalized p[]
		double u = Math.random() * p[K - 1];
		for (topic = 0; topic < K; topic++) {
			if (p[topic] > u) // sample topic w.r.t distribution p
				break;
		}

		// add newly estimated z_i to count variables
		nw[w][topic] += 1;
		nd[m][topic] += 1;
		nwsum[topic] += 1;
		ndsum[m] += 1;

		return topic;
	}

	public void predict_labels(InstanceList instances) {
		for (int d = 0; d < D; d++) {
			FeatureSequence fs = (FeatureSequence) instances.get(d).getData();
			int N = fs.size();
		    for(int n=0; n<N; n++){
		    	assign_z(d, n, fs);
		    }
		}
	}

	private void assign_z(int d, int n, FeatureSequence fs) {
           int best_z = 0;
           double max_prob = 0.0;
           int w = fs.getIndexAtPosition(n);
           for(int k=0; k<K; k++){
        	   if(max_prob < phi[k][w]) {
        		   max_prob = phi[k][w];
        		   best_z = k;
        	   }   
           }
           z[d][n] = best_z;
	}

	public void outputModel(int topK) {
		Alphabet dict = instances.getDataAlphabet();
		if (topK > V) {
			topK = V;
		}
		for (int k = 0; k < K; k++) {
			System.out.println("Topic " + k + "th:");
			HashMap<Integer, Double> map = new HashMap<Integer, Double>();
			for (int w = 0; w < V; w++) {
				double pro = phi[k][w];
				map.put(w, pro); // phi[k][w] compute by nw[w][k], w is the idx
									// of words in dictionary
			}
			LinkedHashMap rankedMap = RankMap.sortHashMapByValues(map, false);
			Set<Integer> keys = rankedMap.keySet();
			Iterator iter = keys.iterator();
			int i = 0;
			while (iter.hasNext() && i < topK) {
				int idx = (Integer) iter.next();
				Object obj = dict.lookupObject(idx);
				double pro = map.get(idx);
				System.out.println("\t" + obj.toString() + " " + pro);
				i++;
			}

		}

	}

	public void writeModel(String outputDir, String fileName) {
		File dir = new File(outputDir);
		FileOutputStream fos;
		try {
			fos = new FileOutputStream(new File(dir, fileName + ".ser"));
			ObjectOutputStream out = new ObjectOutputStream(fos);
			out.writeObject(instances);
			out.writeObject(phi);
			out.writeObject(theta);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public void readModel(String outputDir, String fileName) {
		File dir = new File(outputDir);
		FileInputStream fis;
		ObjectInputStream in = null;
		try {
			fis = new FileInputStream(new File(dir, fileName + ".ser"));
			in = new ObjectInputStream(fis);
			instances = (InstanceList) in.readObject();
			phi = (double[][]) in.readObject();
			theta = (double[][]) in.readObject();

		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
}
