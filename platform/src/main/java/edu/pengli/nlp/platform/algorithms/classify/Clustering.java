package edu.pengli.nlp.platform.algorithms.classify;

import java.util.Arrays;

import edu.pengli.nlp.platform.types.InstanceList;

// data structure to store clusters
public class Clustering {

	public int numLabels;
	protected int[] labels;
	protected InstanceList instances;

	public Clustering(InstanceList instances, int numLabels, int[] labels) {
		this.instances = instances;
		this.numLabels = numLabels;
		this.labels = labels;

	}

	public InstanceList getInstances() {
		return this.instances;
	}

	/** Return an list of instances with a particular label. */
	public InstanceList getCluster(int label) {
		InstanceList cluster = new InstanceList(instances.getPipe());		
		for (int n=0 ; n<instances.size() ; n++) 
	    if (labels[n] == label)
				cluster.add(instances.get(n));			
		return cluster;
	}


	/** Returns an array of instance lists corresponding to clusters. */
	public InstanceList[] getClusters() {
		InstanceList[] clusters = new InstanceList[numLabels];
		for (int c = 0; c < numLabels; c++)
			clusters[c] = getCluster(c);
		return clusters;
	}

	/** Get the cluster label for a particular instance. */
	public int getLabel(int index) {
		return labels[index];
	}

	public int[] getLabels() {
		return labels;
	}

	public int getNumClusters() {
		return numLabels;
	}

	public int getNumInstances() {
		return instances.size();
	}

	public int size(int label) {
		int size = 0;
		for (int i = 0; i < labels.length; i++)
			if (labels[i] == label)
				size++;
		return size;
	}

	public int[] getIndicesWithLabel(int label) {
		int[] indices = new int[size(label)];
		int count = 0;
		for (int i = 0; i < labels.length; i++)
			if (labels[i] == label)
				indices[count++] = i;
		return indices;
	}

	public boolean equals(Object o) {
		Clustering c = (Clustering) o;
		return Arrays.equals(c.getLabels(), labels);
	}

	public void writeClusterResultsToFile() {

	}

	public Clustering shallowCopy() {
		int[] newLabels = new int[labels.length];
		System.arraycopy(labels, 0, newLabels, 0, labels.length);
		Clustering c = new Clustering(instances, numLabels, newLabels);
		return c;
	}

	// SETTERS
	/** Set the cluster label for a particular instance. */
	public void setLabel(int index, int label) {
		labels[index] = label;
	}

	/** Set the number of clusters */
	public void setNumLabels(int n) {
		numLabels = n;
	}

}
