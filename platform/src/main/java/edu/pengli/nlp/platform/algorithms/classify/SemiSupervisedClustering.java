package edu.pengli.nlp.platform.algorithms.classify;

import java.io.IOException;
import java.util.ArrayList;


import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;
import matlabcontrol.extensions.MatlabNumericArray;
import matlabcontrol.extensions.MatlabTypeConverter;
import edu.pengli.nlp.platform.pipe.Pipe;
import edu.pengli.nlp.platform.types.FeatureVector;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.types.Metric;

public class SemiSupervisedClustering extends Clusterer {
	
	MatlabProxy proxy;
	InstanceList seeds;
	Metric metric;

	public SemiSupervisedClustering(Pipe instancePipe, InstanceList seeds, Metric metric,
			MatlabProxy proxy) {

		super(instancePipe);

		this.metric = metric;
		this.seeds = seeds;
		this.proxy = proxy;

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
				weightMatrix[i][j] = metric.distance(fv_i, fv_j);
			}
		}
		int clusterLabels[] = new int[instances.size()];
		try {
			
			processor.setNumericArray("W", new MatlabNumericArray(
					weightMatrix, null));

			double[][] diagonalMatrix = new double[instances.size()+seeds.size()][instances
			                                                      				.size()+seeds.size()];
			
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
//			proxy.eval("q = sum(fl)+1");
//			proxy.eval("fu_CMN = fu.*repmat(q./sum(fu), n-l, 1)");
			
			double[][] fu = processor.getNumericArray("fu").getRealArray2D();
			for(int i=0; i<instances.size(); i++){
				double max = Double.MIN_VALUE;
				int idx = -1;
				for(int j=0; j<seeds.size(); j++){
					if(max <= fu[i][j]){
						max = fu[i][j];
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
