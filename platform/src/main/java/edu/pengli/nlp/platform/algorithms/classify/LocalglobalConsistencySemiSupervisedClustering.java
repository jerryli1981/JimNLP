package edu.pengli.nlp.platform.algorithms.classify;

import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;
import matlabcontrol.extensions.MatlabNumericArray;
import matlabcontrol.extensions.MatlabTypeConverter;
import edu.pengli.nlp.platform.pipe.Pipe;
import edu.pengli.nlp.platform.types.FeatureVector;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.types.Metric;

//Semi-supervised learning with Local and Global Consistency.
public class LocalglobalConsistencySemiSupervisedClustering 
			extends SemiSupervisedClustering{

	public LocalglobalConsistencySemiSupervisedClustering(Pipe instancePipe,
			InstanceList seeds, Metric metric, MatlabProxy proxy) {
		super(instancePipe, seeds, metric, proxy);
	}

	public Clustering cluster(InstanceList instances) {

		MatlabTypeConverter processor = new MatlabTypeConverter(proxy);

		double[][] weightMatrix = new double[instances.size()+seeds.size()][instances
				.size()+seeds.size()];
		InstanceList allInsts = new InstanceList(null);
		for(Instance seed : seeds)
			allInsts.add(seed);
		for(Instance inst : instances)
			allInsts.add(inst);
		

		for (int i = 0; i < allInsts.size(); i++) {
			FeatureVector fv_i = (FeatureVector) allInsts.get(i).getData();
			for (int j = 0; j < allInsts.size(); j++) {
				FeatureVector fv_j = (FeatureVector) allInsts.get(j).getData();
/*				double sum = 0.0;
				for(int k=0; k<fv_i.getValues().length; k++){
					sum += Math.pow((fv_i.getValues()[k]-fv_j.getValues()[k]), 2)/10;
				}*/
//				weightMatrix[i][j] = Math.exp(-sum);
//				weightMatrix[i][j] = 1-metric.distance(fv_i, fv_j);
				weightMatrix[i][j] = metric.distance(fv_i, fv_j);
			}
		}
		int clusterLabels[] = new int[instances.size()];
		try {
			
			
			processor.setNumericArray("W", new MatlabNumericArray(
					weightMatrix, null));

			
			double[][] Y = new double[instances.size()+seeds.size()][instances.size()+seeds.size()];
			for (int i = 0; i < seeds.size(); i++) {
				for (int j = 0; j < seeds.size(); j++) {
						if(i == j)
							Y[i][j] = 1.0;
						else
							Y[i][j] = 0.0;
				}
			}

			processor.setNumericArray("Y", new MatlabNumericArray(
					Y, null));
			proxy.eval("alpha = 0.3");
			proxy.eval("n = size(W,1)");		
			proxy.eval("I = eye(n,n)");
			proxy.eval("W  = W -diag(diag(W))");
			proxy.eval("D = sum(W,2)");
			proxy.eval("D12 = diag(1 ./sqrt(D));");
			proxy.eval("S = D12 * W * D12");
			proxy.eval("beta = 1 - alpha");
			proxy.eval("p = 5");
			proxy.eval("M = I - (alpha * S)");
			proxy.eval("T = mldivide(M,Y)");
			proxy.eval("F = beta*T");
			
			double[][] F = processor.getNumericArray("F").getRealArray2D();
			for(int i=0; i<instances.size(); i++){
				double max = Double.MIN_VALUE;
				int idx = -1;
				for(int j=0; j<seeds.size(); j++){
					if(max <= F[i+seeds.size()][j]){
						max = F[i+seeds.size()][j];
						idx = j;
					}
				}
				clusterLabels[i] = idx;	
			}
							
		} catch (MatlabInvocationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return new Clustering(instances, seeds.size(), clusterLabels);

	}
}
