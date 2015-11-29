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

public class FCALGreens_Cos extends SemiSupervisedClustering{

	int sigma = 0;
	
	public FCALGreens_Cos(Pipe instancePipe,
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
/*				double sum = 0.0;
				for(int k=0; k<fv_i.getValues().length; k++){
					sum += Math.pow((fv_i.getValues()[k]-fv_j.getValues()[k]), 2)/sigma;
				}
				weightMatrix[i][j] = Math.exp(-sum);*/
				weightMatrix[i][j] = 1-metric.distance(fv_i, fv_j);
			}
		}
		int clusterLabels[] = new int[instances.size()];
		try {
					
			processor.setNumericArray("W", new MatlabNumericArray(
					weightMatrix, null));
		
			proxy.eval("D = diag(sum(W,2))");
			proxy.eval("Dd = diag(D)");		
			proxy.eval("Dn=diag(sqrt(1./Dd))");
			proxy.eval("Wn = Dn*W*Dn");
			proxy.eval("[v, s] = eig(full(diag(sum(Wn)) - Wn))");
			proxy.eval("num_eig_vectors = size(Wn, 1) - 1");
			proxy.eval("s = diag(s)");
			proxy.eval("m_idx=find(s<10^-3)");
			proxy.eval("idx_using = setdiff(1 : num_eig_vectors + 1, m_idx)");
			proxy.eval("s_using = 1 ./ s(idx_using)");
			proxy.eval("v_using = v(:, idx_using)");
			proxy.eval("greens_matrix = zeros(size(Wn))");
			proxy.eval("iterTime = length(s_using)");
			int iterTime = (int)processor.getNumericArray("iterTime").getRealValue(0);
			for(int k = 1; k <= iterTime; k++){
				proxy.eval("v_in_use = v_using(:, "+k+")");
				proxy.eval("greens_matrix = greens_matrix + s_using("+k+") * v_in_use * v_in_use'");
			}
			
			double[][] inputMatrix = new double[instances.size()+seeds.size()][seeds.size()];
			for (int i = 0; i < instances.size()+seeds.size(); i++) {
				for (int j = 0; j < seeds.size(); j++) {
						if(i == j)
							inputMatrix[i][j] = 1.0;
						else
							inputMatrix[i][j] = 0.0;
				}
			}
				
			processor.setNumericArray("input_matrix", new MatlabNumericArray(
					inputMatrix, null));
			proxy.eval("input_matrix = input_matrix");
			int nl = seeds.size();
			proxy.eval("idx_train = (1:"+nl+")'");
			proxy.eval("idx_test = ("+nl+"+1:"+allInsts.size()+")'");
			proxy.eval("input_matrix(idx_test, :) = 0");
			proxy.eval("n_input_matrix_pos = sum(input_matrix, 1)");
			proxy.eval("n_input_matrix_neg = length(idx_train) - n_input_matrix_pos");
			proxy.eval("iterTime = size(input_matrix, 2)");
			iterTime = (int)processor.getNumericArray("iterTime").getRealValue(0);
			for(int k = 1; k <= iterTime; k++){
				proxy.eval("idx_pos = intersect(find(input_matrix(:, "+k+") ~= 0), find(idx_train))");
				proxy.eval("idx_neg = intersect(find(input_matrix(:, "+k+") == 0), find(idx_train))");
				proxy.eval("input_matrix(idx_pos, "+k+") = 1 / n_input_matrix_pos("+k+")");
				proxy.eval("input_matrix(idx_neg, "+k+") = -1 / n_input_matrix_neg("+k+")");
			}
							
			double[][] Y = new double[seeds.size()][seeds.size()];
			for (int i = 0; i < seeds.size(); i++) {
				for (int j = 0; j < seeds.size(); j++) {
						if(i == j)
							Y[i][j] = 1.0;
						else
							Y[i][j] = 0.0;
				}
			}
			

			processor.setNumericArray("label_correlation_matrix", new MatlabNumericArray(
					Y, null));
			
			proxy.eval("F = greens_matrix * input_matrix * inv(eye(size(label_correlation_matrix)) -"
					+ " 0.1 * label_correlation_matrix)");
						
			double[][] F = processor.getNumericArray("F").getRealArray2D();
			for(int i=0; i<instances.size(); i++){
				double max = Double.NEGATIVE_INFINITY;
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

