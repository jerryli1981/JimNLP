package edu.pengli.nlp.conference.acl2015.pipe;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Stack;

import edu.pengli.nlp.conference.acl2015.types.Argument;
import edu.pengli.nlp.conference.acl2015.types.Predicate;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.BasicDependenciesAnnotation;
import edu.stanford.nlp.trees.CollinsHeadFinder;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.trees.HeadFinder;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.util.CoreMap;

public class HeadAnnotation {

	static HeadFinder headFinder;

	public HeadAnnotation() {

		headFinder = new CollinsHeadFinder();
	}

	// return value could be null
	public Predicate annotatePredicateHead(Predicate pred, CoreMap sent) {

		if (pred.size() == 1) {
			pred.setHead(pred.get(0));
			return pred;
		}

		boolean find = false;
		SemanticGraph graph = sent.get(BasicDependenciesAnnotation.class);
		Stack<Integer> stack = new Stack<Integer>();
		boolean[] marked = new boolean[graph.size() * 2];
		int rootIdx = graph.getFirstRoot().index();
		marked[rootIdx] = true;
		stack.add(rootIdx);
		while (!stack.isEmpty()) {
			int s = stack.pop();
			Iterable<SemanticGraphEdge> iter = graph.outgoingEdgeIterable(graph
					.getNodeByIndex(s));
			for (SemanticGraphEdge edge : iter) {
				GrammaticalRelation gr = edge.getRelation();
				IndexedWord gov = edge.getGovernor();
				if (gr.toString().equals("nsubj")
						|| gr.toString().equals("dobj")
						|| (gr.toString().equals("prep") && gov.tag()
								.startsWith("VB"))) {

					for (IndexedWord tok : pred) {
						int preCoreLabelIdx = tok.index();
						IndexedWord preiw = graph
								.getNodeByIndexSafe(preCoreLabelIdx);
						if (preiw == null)
							continue;
						if (preiw.equals(edge.getGovernor())) {
							pred.setHead(preiw);
							find = true;
						}
					}
				}

				int depIdx = edge.getDependent().index();
				if (!marked[depIdx]) {
					marked[depIdx] = true;
					stack.add(depIdx);
				}
			}
		}

		if (find == false) {
			for (IndexedWord cl : pred) {
				if (cl.tag().startsWith("VB")) {
					pred.setHead(cl);
				}
			}
		}

		return pred;
	}

	public Argument annotateArgHead(Argument arg, CoreMap sent) {

		if (arg.size() == 1) {
			arg.setHead(arg.get(0));
			return arg;
		}

		Tree tree = sent.get(TreeAnnotation.class);
		HashMap<String, String> npheadMap = new HashMap<String, String>();
		dfs(tree, tree, headFinder, npheadMap);

		String argMention = arg.originaltext();
		boolean find = false;

		for (String np : npheadMap.keySet()) {
			if (npheadMap.containsKey(argMention)) {
				find = true;
				String headMention = npheadMap.get(np);
				for (IndexedWord lab : arg) {
					if (lab.originalText().equals(headMention)) {
						arg.setHead(lab);
					}
				}
			}
		}

		if (find == true)
			return arg;

		SemanticGraph graph = sent.get(BasicDependenciesAnnotation.class);

		List<SemanticGraphEdge> path = new ArrayList<SemanticGraphEdge>();
		for (int i = 0; i < arg.size(); i++) {
			for (int j = 0; j < arg.size(); j++) {
				if (i == j)
					continue;
				else {
					IndexedWord ai = arg.get(i);
					IndexedWord aj = arg.get(j);
					if (ai == null || aj == null)
						continue;
					List<SemanticGraphEdge> edge = graph.getAllEdges(ai, aj);
					path.addAll(edge);
				}
			}
		}

		// if arguments contains complicated structures, just ignore it to keep
		// readability
		for (SemanticGraphEdge edge : path) {
			GrammaticalRelation gr = edge.getRelation();
			if (gr.toString().equals("nsubj") || gr.toString().equals("ccomp")
					|| gr.toString().equals("nsubjpass")) {
				return null;
			}
		}

		return decideArgHead(arg, path);

	}

	private Argument decideArgHead(Argument arg, List<SemanticGraphEdge> path) {

		int startNodeIdx = arg.get(0).index();
		int endNodeIdx = arg.get(arg.size() - 1).index();

		boolean find = false;
		for (SemanticGraphEdge edge : path) {
			GrammaticalRelation gr = edge.getRelation();
			if (gr.toString().equals("det") || gr.toString().equals("amod")
					|| gr.toString().equals("poss")
					|| gr.toString().equals("num")
					|| gr.toString().equals("advmod")
					|| gr.toString().equals("prep")) {
				IndexedWord gov = edge.getGovernor();
				if (!gov.tag().startsWith("NN"))
					continue;
				if (gov.index() < startNodeIdx || gov.index() > endNodeIdx)
					continue;

				arg.setHead(gov);
				find = true;
			} else if (gr.toString().equals("tmod")
					|| gr.toString().equals("nn")) {
				IndexedWord dep = edge.getDependent();
				if (!dep.tag().startsWith("NN"))
					continue;
				if (dep.index() < startNodeIdx || dep.index() > endNodeIdx)
					continue;

				arg.setHead(dep);
				find = true;
			}
		}

		if (find == true)
			return arg;
		else {
			for (int i = 0; i < arg.size(); i++) {
				if (arg.get(i).tag().equals("NNS")) {
					arg.setHead(arg.get(i));
				}
			}
		}

		return arg;
	}

	public void dfs(Tree node, Tree parent, HeadFinder headFinder,
			HashMap<String, String> map) {
		if (node == null || node.isLeaf()) {
			return;
		}
		// if node is a NP - Get the terminal nodes to get the words in the NP
		if (node.value().equals("NP")) {

			// System.out.println(" Noun Phrase is ");
			List<Tree> leaves = node.getLeaves();
			StringBuilder np = new StringBuilder();
			for (Tree leaf : leaves) {
				// System.out.print(leaf.toString()+" ");
				np.append(leaf.toString() + " ");
			}

			// System.out.println();
			// System.out.println(" Head string is ");
			// System.out.println(node.headTerminal(headFinder, parent));
			String head = node.headTerminal(headFinder, parent).toString();
			String nounPhrase = np.toString().trim();
			nounPhrase = nounPhrase.replaceAll("\\s,", ",");
			nounPhrase = nounPhrase.replaceAll(" '", "'");

			map.put(nounPhrase, head);
		}

		for (Tree child : node.children()) {
			dfs(child, node, headFinder, map);
		}
	}

}
