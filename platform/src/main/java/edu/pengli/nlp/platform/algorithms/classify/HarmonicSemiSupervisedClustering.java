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


//Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions
public class HarmonicSemiSupervisedClustering extends SemiSupervisedClustering{
	
	int sigma = 0;
	
	public HarmonicSemiSupervisedClustering(Pipe instancePipe,
			InstanceList seeds, Metric metric, MatlabProxy proxy, int sigma) {
		super(instancePipe, seeds, metric, proxy);
		this.sigma = sigma;
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
				double sum = 0.0;
				for(int k=0; k<fv_i.getValues().length; k++){
					sum += Math.pow((fv_i.getValues()[k]-fv_j.getValues()[k]), 2)/sigma;
				}
				weightMatrix[i][j] = Math.exp(-sum);
//				weightMatrix[i][j] = 1-metric.distance(fv_i, fv_j);
			}
		}
		int clusterLabels[] = new int[instances.size()];
		try {
			
			
			processor.setNumericArray("W", new MatlabNumericArray(
					weightMatrix, null));


			double[][] diagonalMatrix = new double[instances.size()+
			                                       seeds.size()][instances.size()+seeds.size()];
			
			for (int i = 0; i < allInsts.size(); i++) {
				double sum = 0.0;
				for (int j = 0; j < allInsts.size(); j++) {
					sum += weightMatrix[i][j];
				}
				diagonalMatrix[i][i] = sum;
			}
			
			processor.setNumericArray("D", new MatlabNumericArray(
					diagonalMatrix, null));
			
			double[][] fl = new double[seeds.size()][seeds.size()];
			
			for (int i = 0; i < seeds.size(); i++) {
				for (int j = 0; j < seeds.size(); j++) {
						if(i == j)
							fl[i][j] = 1.0;
						else
							fl[i][j] = 0.0;
				}
			}

			processor.setNumericArray("fl", new MatlabNumericArray(
					fl, null));
			proxy.eval("L=D-W");		
			proxy.eval("l = size(fl, 1)");
			proxy.eval("n = size(L, 1)");
			proxy.eval("fu = -inv(L(l+1:n, l+1:n)) * L(l+1:n, 1:l) * fl");
			proxy.eval("q = sum(fl)+1");
			proxy.eval("fu_CMN = fu.*repmat(q./sum(fu), n-l, 1)");
			
			double[][] fu_CMN = processor.getNumericArray("fu_CMN").getRealArray2D();
			for(int i=0; i<instances.size(); i++){
				double max = Double.MIN_VALUE;
				int idx = -1;
				for(int j=0; j<seeds.size(); j++){
					if(max <= fu_CMN[i][j]){
						max = fu_CMN[i][j];
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
