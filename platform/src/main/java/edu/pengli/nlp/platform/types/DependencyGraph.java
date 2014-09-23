package edu.pengli.nlp.platform.types;

import java.util.LinkedList;

import edu.stanford.nlp.trees.TypedDependency;

public class DependencyGraph {

	private int V; // number of vertices
	private int E; // number of edges
	private LinkedList<TypedDependency>[] adj;

	public DependencyGraph(int V) {
		this.V = V;
		this.E = 0;
		adj = (LinkedList<TypedDependency>[]) new LinkedList[V];
		for (int v = 0; v < V; v++)
			adj[v] = new LinkedList<TypedDependency>();
	}

	public int V() {
		return V;
	}

	public int E() {
		return E;
	}

	public void addEdge(TypedDependency e) {
		adj[e.gov().index()].add(e);
		E++;
	}

	public Iterable<TypedDependency> adj(int v) {
		return adj[v];
	}

}
