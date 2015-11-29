package edu.pengli.nlp.platform.algorithms;

import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;
import matlabcontrol.extensions.MatlabNumericArray;
import matlabcontrol.extensions.MatlabTypeConverter;
import edu.pengli.nlp.platform.algorithms.classify.Clustering;
import edu.pengli.nlp.platform.types.FeatureVector;
import edu.pengli.nlp.platform.types.InstanceList;

public class ClustererUtil {
	
	public static int getNumberOfClusters(InstanceList instances, MatlabProxy proxy, double thPer) {
		
		MatlabTypeConverter processor = new MatlabTypeConverter(proxy);
		
		FeatureVector vec = (FeatureVector)instances.get(0).getData();
		int dimension = vec.getValues().length;
		
		double[][] dataMatrix = new double[instances.size()][dimension];
		for (int i = 0; i < instances.size(); i++) {
			FeatureVector fv_i = (FeatureVector) instances.get(i).getData();
			for(int j=0; j<dimension; j++)
				dataMatrix[i][j] = fv_i.getValues()[j];
		}
		int numC = -1;
		try {
			
			processor.setNumericArray("arr", new MatlabNumericArray(
					dataMatrix, null));
			proxy.eval("X = arr'");
			proxy.eval("addpath('/home/peng/Downloads/groupNumberMatlabCode')");
			proxy.eval("numC = groupNumber(X, "+thPer+")");
			numC = (int)processor.getNumericArray("numC").getRealValue(0);
/*			proxy.eval("X = arr'");
			proxy.eval("percent = 0.01");
			proxy.eval("[dim n] = size(X)");
			
			proxy.eval("tmp = zeros(n, n);");
			
			for(int ii=1; ii<=instances.size(); ii++){
				for(int jj=1; jj<=instances.size(); jj++){
					proxy.eval("diff = X(:, "+ii+") - X(:, "+jj+")");
					proxy.eval("tmp("+ii+", "+jj+") = sqrt(sum(diff .* diff))");
				}
			}
			
			proxy.eval("sortT = sort(tmp, 2)");
			proxy.eval("aveT = sum(sortT)/n");
			proxy.eval("threshold = aveT(floor(n*percent))");
			double threshold = processor.getNumericArray("threshold").getRealValue(0);
			proxy.eval("den = zeros(n, 1)");
			proxy.eval("dist = zeros(n, 1)");
			
			for(int ii=1; ii<=instances.size(); ii++){
				for(int jj=1; jj<=instances.size(); jj++){
					proxy.eval("t = tmp("+ii+", "+jj+")");
					double t = processor.getNumericArray("t").getRealValue(0);
					if(t < threshold && ii != jj){
						proxy.eval("den("+ii+") = den("+ii+") + 1");
					}
				}
			}
			
			for(int ii=1; ii<=instances.size(); ii++){
				proxy.eval("tmpDist = max(max(tmp))");
				double tmpDist = processor.getNumericArray("tmpDist").getRealValue(0);
				for(int jj=1; jj<=instances.size(); jj++){
					proxy.eval("t = tmp("+ii+", "+jj+")");
					double t = processor.getNumericArray("t").getRealValue(0);
					proxy.eval("j = den("+jj+")");
					double j = processor.getNumericArray("j").getRealValue(0);
					proxy.eval("i = den("+ii+")");
					double i = processor.getNumericArray("i").getRealValue(0);
					if(j > i && t < tmpDist){
						proxy.eval("tmpDist = tmp("+ii+", "+jj+")");
					}
				}
			}
			
			proxy.eval("gamma = den .* dist");
			proxy.eval("thre = sum(gamma) * "+thPer);
			proxy.eval("numC = length(find(gamma > thre))");
			numC = (int)processor.getNumericArray("numC").getRealValue(0);*/
			
		} catch (MatlabInvocationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return numC;
	}

}
