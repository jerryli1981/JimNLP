package edu.pengli.nlp.platform.algorithms.classify;

import java.io.IOException;
import java.util.ArrayList;

import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLDouble;

import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;
import matlabcontrol.extensions.MatlabNumericArray;
import matlabcontrol.extensions.MatlabTypeConverter;
import edu.pengli.nlp.platform.pipe.Pipe;
import edu.pengli.nlp.platform.types.FeatureVector;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.types.Metric;

public class GraphCut_ALM2 extends Clusterer {

	

	Metric metric;
	int numClusters;
	MatlabProxy proxy;

	public GraphCut_ALM2(Pipe instancePipe, int numClusters, Metric metric,
			MatlabProxy proxy) {

		super(instancePipe);

		this.metric = metric;
		this.numClusters = numClusters;
		this.proxy = proxy;

	}

	public Clustering cluster(InstanceList instances) {

		MatlabTypeConverter processor = new MatlabTypeConverter(proxy);

		double[][] similarityMatrix = new double[instances.size()][instances
				.size()];

		for (int i = 0; i < instances.size(); i++) {
			FeatureVector fv_i = (FeatureVector) instances.get(i).getData();
			for (int j = 0; j < instances.size(); j++) {
				FeatureVector fv_j = (FeatureVector) instances.get(j).getData();

/*				double sum = 0.0;
				for (int k = 0; k < fv_i.getValues().length; k++) {
					sum += Math.pow(
							(fv_i.getValues()[k] - fv_j.getValues()[k]), 2) / 10;
				}

				similarityMatrix[i][j] = Math.exp(-sum);*/
				similarityMatrix[i][j] = 1 - metric.distance(fv_i, fv_j);
			}
		}
		/*		
		ArrayList list = new ArrayList();
		double[] arr = new double[instances.size()*instances.size()];
		int c = 0;
		for(int i=0; i<instances.size(); i++){
			for(int j=0; j<instances.size(); j++)
				arr[c++] = similarityMatrix[i][j];
		}
		
		ArrayList<MLDouble> ret = new ArrayList<MLDouble>();
		ret.add(new MLDouble("Matrix", arr, instances.size()));
		list.addAll(ret);
		
		String matInputFile = "/home/peng/Downloads/GraphCutInput2.mat";
		try {
			new MatFileWriter(matInputFile, list);
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}*/
		
		int clusterLabels[] = new int[instances.size()];
		int NITER = 150;

		try {
			processor.setNumericArray("S", new MatlabNumericArray(
					similarityMatrix, null));

			double mu = 0.1;
			double rho = 1.5;

			proxy.eval("obj = zeros(" + NITER + ",1)");
			proxy.eval("n = size(S, 1)");
			proxy.eval("Lambda = zeros(n)");
			proxy.eval("Sigma = zeros(n)");
			proxy.eval("one = ones(n, 1)");
			proxy.eval("In = eye(n)");

			proxy.eval("B = S");
			proxy.eval("A = S");
			proxy.eval("D = diag(sum(A, 2))");
			proxy.eval("L = D - A");


/*			proxy.eval("fl = full(L)");
			proxy.eval("fl = max(fl,fl')");
			proxy.eval("[v d] = eig(fl)");
			proxy.eval("d = diag(d)");
			proxy.eval("[d1, idx] = sort(d)");
			proxy.eval("idx1 = idx(1:" + numClusters + ")");
			proxy.eval("a = d(idx)");

			proxy.eval("x = sum(a(1:" + numClusters + "+1))");
			double x = processor.getNumericArray("x").getRealValue(0);
			if (x < EPSILON) {
				System.err.println("The original graph has more than "
						+ numClusters + " connected component");
			}*/

			for (int iter = 1; iter <= NITER; iter++) {
				proxy.eval("inmu = 1 /" + mu);
				proxy.eval(" M = B - inmu*Sigma - L + inmu*Lambda + diag(L-inmu*Lambda)*one'");
				proxy.eval("diagA =  diag(B - inmu*Sigma)");
				proxy.eval("A1 = 1/(n+1) * M * one + n/(n+1) * diagA");
				proxy.eval("A = (M + diagA*one' + diag(A1)) * (0.5*In - 1/(2*n+4)*one*one')");
				proxy.eval("D = diag(sum(A,2))");
				proxy.eval("T = A + inmu*Sigma");
				proxy.eval("BB = (S + S' + (T+T')*" + mu + "/2) / (" + mu
						+ "+2)");
				proxy.eval("B = zeros(n)");
				proxy.eval("B(find(BB > 0)) = BB(find(BB>0))");

				proxy.eval("[U, SS, V] = svd(D-A+inmu*Lambda)");
				proxy.eval("[SS_sort, SS_index] = sort(diag(SS), 'descend')");
				proxy.eval("U = U(:, SS_index(1: n-" + numClusters + "))");
				proxy.eval("V = V(:, SS_index(1: n-" + numClusters + "))");
				proxy.eval("SS = SS(SS_index(1: n-" + numClusters
						+ "), SS_index(1: n-" + numClusters + "))");
				proxy.eval("L = U * SS * V'");
				proxy.eval("Lambda = Lambda + " + mu + "*(D-A-L)");
				
				proxy.eval("Sigma = Sigma + " + mu + "*(A-B)");
				double Sigma = processor.getNumericArray("Sigma").getRealValue(0);
//				System.out.println("Sigma "+ Sigma);
				proxy.eval("mu = min(10^10," + rho * mu + ")");
				proxy.eval("obj(" + iter + ") = sum(sum((B-S).*(B-S)))");
			}
			
/*			double[][] obj = processor.getNumericArray("obj").getRealArray2D();
			double[] arr = new double[processor.getNumericArray("obj").getLength()];
			for(int i=0; i<processor.getNumericArray("obj").getLength(); i++)
				arr[i] = obj[i][0];
					
			
			ArrayList list= new ArrayList(); 
			list.add(new MLDouble("obj", arr, processor.getNumericArray("obj").getLength()));
			String matInputFile = "/home/peng/Downloads/visual/obj.mat";
			 try { 
				 new MatFileWriter(matInputFile, list); 
			} catch(IOException e) { // TODO Auto-generated catch block
				 e.printStackTrace(); 
			}*/

			proxy.eval("fl = full(L)");
			proxy.eval("fl = max(fl,fl')");
			proxy.eval("[v d] = eig(fl)");
			proxy.eval("d = diag(d)");
			proxy.eval("[d1, idx] = sort(d)");
			proxy.eval("idx1 = idx(1:" + numClusters + ")");
			proxy.eval("F = v(:,idx1)");
			proxy.eval("[labv, tem, Ind] = unique(round(0.1*round(1000*F)),'rows')");
			proxy.eval("BB = (B+B')/2");
			proxy.eval("LL = diag(sum(BB, 2)) - BB");

			proxy.eval("fll = full(LL)");
			proxy.eval("fll = max(fll,fll')");
			proxy.eval("[v d] = eig(fll)");
			proxy.eval("d = diag(d)");
			proxy.eval("[d1, idx] = sort(d)");
			proxy.eval("idx1 = idx(1:" + numClusters + ")");
			proxy.eval("FF = v(:,idx1)");

			proxy.eval("labels = kmeans(FF," + numClusters
					+ ",'Replicates',20)");

			double[][] labels = processor.getNumericArray("labels")
					.getRealArray2D();

			for (int i = 0; i < instances.size(); i++)
				clusterLabels[i] = (int) labels[i][0] - 1;

		} catch (MatlabInvocationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return new Clustering(instances, numClusters, clusterLabels);
	}

}
