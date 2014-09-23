package edu.pengli.nlp.platform.pipe;

import edu.pengli.nlp.platform.types.FeatureCounter;
import edu.pengli.nlp.platform.types.FeatureSequence;
import edu.pengli.nlp.platform.types.FeatureVector;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import gnu.trove.map.hash.TIntIntHashMap;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

public class FeatureIDFPipe extends Pipe {

	FeatureCounter counter;
	int numDocs;

	public FeatureIDFPipe(InstanceList tf_fvs) {

		counter = new FeatureCounter(tf_fvs.getDataAlphabet());

		HashMap<String, ArrayList<FeatureVector>> map = new HashMap<String, ArrayList<FeatureVector>>();
		for (int i = 0; i < tf_fvs.size(); i++) {
			Instance inst = tf_fvs.get(i);
			FeatureVector fv = (FeatureVector) inst.getData();
			String name = (String) inst.getName();
			String docName = name.split("_")[1];
			if (!map.containsKey(docName)) {
				ArrayList<FeatureVector> list = new ArrayList<FeatureVector>();
				list.add(fv);
				map.put(docName, list);
			} else {
				ArrayList<FeatureVector> list = map.get(docName);
				list.add(fv);
				map.put(docName, list);
			}
		}

		numDocs = map.keySet().size();
		ArrayList<FeatureVector> docFVs = new ArrayList<FeatureVector>();
		Set<String> keys = map.keySet();
		Iterator<String> iter = keys.iterator();
		while (iter.hasNext()) {
			String key = iter.next();
			ArrayList<FeatureVector> val = map.get(key);
			ArrayList<Integer> indexs = new ArrayList<Integer>();
			ArrayList<Double> vals = new ArrayList<Double>();
			for (FeatureVector fv : val) {
				int[] idx = fv.getIndices();
				double[] values = fv.getValues();
				for (int i = 0; i < idx.length; i++) {
					indexs.add(idx[i]);
					vals.add(values[i]);
				}
			}
			int[] INDEX = new int[indexs.size()];
			double[] VALS = new double[vals.size()];
			for(int i=0; i<indexs.size(); i++){
				INDEX[i] = indexs.get(i);
				VALS[i] = vals.get(i);
			}
			docFVs.add(new FeatureVector(INDEX, VALS));
		}

		for (FeatureVector docfv : docFVs) {
			TIntIntHashMap localCounter = new TIntIntHashMap();
			int[] index = docfv.getIndices();
			for (int i = 0; i < index.length; i++) {
				localCounter.adjustOrPutValue(index[i], 1, 1);
			}
			for (int feature : localCounter.keys()) {
				counter.increment(feature);
			}
		}
	}

	protected Instance pipe(Instance inst) {

		FeatureVector tf_fv = (FeatureVector) inst.getData();
		int[] indexs = tf_fv.getIndices();
		double[] vals = new double[indexs.length];
		for (int i = 0; i < indexs.length; i++) {
			int idx = indexs[i];
			double df = counter.get(idx);
			double idf = Math.log10(numDocs / (df));
			vals[i] = idf;
		}

		FeatureVector idf_fv = new FeatureVector(indexs, vals);
		FeatureVector[] tf_idf_fv = new FeatureVector[2];
		tf_idf_fv[0] = tf_fv;
		tf_idf_fv[1] = idf_fv;

		return new Instance(tf_idf_fv, inst.getTarget(), inst.getName(), inst.getSource());
	}

}
