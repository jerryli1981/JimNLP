package edu.pengli.nlp.conference.acl2015.pipe;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.TreeMap;

import edu.pengli.nlp.conference.acl2015.types.Argument;
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

public class HeadExtractor {
	
	static HeadFinder headFinder;
	
	public HeadExtractor(){
		headFinder = new CollinsHeadFinder();
	}
			
	public Argument extract(Argument arg, CoreMap sent, String type){
		
		if(arg.size() == 1)
			return arg;
		
		Tree tree = sent.get(TreeAnnotation.class);
		HashMap<String, String> npheadMap = new HashMap<String, String>();
		dfs(tree, tree, headFinder, npheadMap);

/*		HashMap<String, Integer> npSizeMap = new HashMap<String, Integer>();
		for (String np : npheadMap.keySet()) {
			npSizeMap.put(np, np.split(" ").length);
		}

		LinkedHashMap<String, Integer> rankedNpSizeMap = RankMap
				.sortHashMapByValues(npSizeMap, false);

		ArrayList<String> npList = new ArrayList<String>();
		for (String np : rankedNpSizeMap.keySet()) {
			npList.add(np);
		}
		ArrayList<Integer> rmIdxList = new ArrayList<Integer>();
		for (int i = 0; i < npList.size() - 1; i++) {
			String np_i = npList.get(i);
			for (int j = i + 1; j < npList.size(); j++) {
				String np_j = npList.get(j);
				if (np_i.contains(np_j) && !rmIdxList.contains(j)) {
					rmIdxList.add(j);
				}
			}
		}
		
		ArrayList<String> mergedNpList = new ArrayList<String>();
		for (int i = 0; i < npList.size(); i++) {
			if (!rmIdxList.contains(i))
				mergedNpList.add(npList.get(i));
		}*/
		
		String argMention = arg.toString();
		boolean find = false;
		for(String np : npheadMap.keySet()){
			if(np.equals(argMention)){
				find  = true;
				String headMention = npheadMap.get(np);
				for(CoreLabel lab : arg){
					if(lab.originalText().equals(headMention)){
						Argument head = new Argument();
						head.add(lab);
						return head;
					}
				}
			}
		}
		
		if(find == false){
			
			if(sent.toString().contains("The attack on West Nickel Mines Amish School")){
				System.out.println();
				System.out.println();
			}
			
			if(type == "1"){
				
				return decideHeadFromSemanticGraph1(arg, sent);
				
			}else if(type == "2"){
				
				return decideHeadFromSemanticGraph2(arg, sent);
				
			}
				
		}
		
		return null;
	
	}
	
	private int getIndex(Argument arg, IndexedWord tok){
		for(int i=0; i<arg.size(); i++){
			if(arg.get(i).originalText().equals(tok.originalText()))
				return i;
		}
		return -1;
	}
	private Argument decideHeadFromSemanticGraph1(Argument arg,
			CoreMap sent) {
			
		SemanticGraph graph = sent.get(BasicDependenciesAnnotation.class);
		
		IndexedWord startNode = graph.getNodeByIndex(arg.get(0).index());
		IndexedWord endNode = graph.getNodeByIndex(arg.get(arg.size()-1).index());
		
		if(startNode.index() > endNode.index()){
			int flagIdx = arg.get(1).index();
			List<IndexedWord> dups = graph.getAllNodesByWordPattern(arg.get(0).originalText());
			TreeMap<Integer, IndexedWord> map = new TreeMap<Integer, IndexedWord>();
			for(IndexedWord w : dups){
				map.put(w.index(), w);
			}
			for(Integer i : map.keySet()){
				if(i < flagIdx){
					startNode = map.get(i);
				}
			}
		}

		
		List<SemanticGraphEdge> allPath = new ArrayList<SemanticGraphEdge>();
		ArrayList<IndexedWord[] > pairs = new ArrayList<IndexedWord[]>();
		
		for(int i=0; i<arg.size(); i++){
			for(int j=0; j<arg.size(); j++){
				if(i == j)
					continue;
				else{
					IndexedWord[] pair = new IndexedWord[2];
					IndexedWord ai = graph.getNodeByIndex(arg.get(i).index());
					IndexedWord aj = graph.getNodeByIndex(arg.get(j).index());
					pair[0] = ai;
					pair[1] = aj;
					pairs.add(pair);
				}
			}
		}
		for(IndexedWord[] pa : pairs){
			allPath.addAll(graph.getAllEdges(pa[0], pa[1]));
		}
		boolean containsSubjectVerb = false;
		for(SemanticGraphEdge edge : allPath){
			GrammaticalRelation gr = edge.getRelation();
			if(gr.toString().equals("nsubj") || gr.toString().equals("ccomp")
					|| gr.toString().equals("nsubjpass") || gr.toString().equals("dobj")){
				IndexedWord gov = edge.getGovernor();
				if(gov.index() < startNode.index() || gov.index() > endNode.index())
					continue;
				containsSubjectVerb = true;
				continue;
			}
			
		}
		
		if(containsSubjectVerb == true)
			return null;
		
		boolean find = false;	
		
		for(SemanticGraphEdge edge : allPath){
			GrammaticalRelation gr = edge.getRelation();
			if(gr.toString().equals("prep")){
				IndexedWord dep = edge.getDependent();
				if(!dep.equals(startNode))
					endNode = dep;
			}
			
		}
		
		
		for(SemanticGraphEdge edge : allPath){
			GrammaticalRelation gr = edge.getRelation();

			
			if (gr.toString().equals("det") || gr.toString().equals("num") || gr.toString().equals("poss")
					|| gr.toString().equals("amod") || gr.toString().equals("advmod")) {
				IndexedWord gov = edge.getGovernor();
				if(!gov.tag().startsWith("NN")) 
					continue;
				if(gov.index() < startNode.index() || gov.index() > endNode.index())
					continue;
				Argument head = new Argument();
				if(getIndex(arg, gov) == -1){
					continue;
				}
				head.add(arg.get(getIndex(arg, gov)));
				find  = true;
				return head;	
			}else if(gr.toString().equals("tmod")){
				IndexedWord dep = edge.getDependent();
				if(!dep.tag().startsWith("NN")) 
					continue;
				if(dep.index() < startNode.index() || dep.index() > endNode.index())
					continue;
				Argument head = new Argument();
				if(getIndex(arg, dep) == -1){
					continue;
				}
				head.add(arg.get(getIndex(arg, dep)));
				find  = true;
				return head;
				
			}
		}
		
		if(find == false){
			for(int i=0; i<arg.size(); i++){
				if(arg.get(i).tag().equals("NNS")){
					Argument head = new Argument();
					head.add(arg.get(i));
					return head;
					
				}
			}
			
		}
		
		return null;
	}
	
	private Argument decideHeadFromSemanticGraph2(Argument arg,
			CoreMap sent) {
				
		SemanticGraph graph = sent.get(BasicDependenciesAnnotation.class);
		
		IndexedWord startNode = graph.getNodeByIndex(arg.get(0).index());
		IndexedWord endNode = graph.getNodeByIndex(arg.get(arg.size()-1).index());
		
		if(startNode.index() > endNode.index()){
			int flagIdx = arg.get(1).index();
			List<IndexedWord> dups = graph.getAllNodesByWordPattern(arg.get(0).originalText());
			TreeMap<Integer, IndexedWord> map = new TreeMap<Integer, IndexedWord>();
			for(IndexedWord w : dups){
				map.put(w.index(), w);
			}
			for(Integer i : map.keySet()){
				if(i < flagIdx){
					startNode = map.get(i);
				}
			}
		}

		
		List<SemanticGraphEdge> allPath = new ArrayList<SemanticGraphEdge>();
		ArrayList<IndexedWord[] > pairs = new ArrayList<IndexedWord[]>();
		
		for(int i=0; i<arg.size(); i++){
			for(int j=0; j<arg.size(); j++){
				if(i == j)
					continue;
				else{
					IndexedWord[] pair = new IndexedWord[2];
					IndexedWord ai = graph.getNodeByIndex(arg.get(i).index());
					IndexedWord aj = graph.getNodeByIndex(arg.get(j).index());
					pair[0] = ai;
					pair[1] = aj;
					pairs.add(pair);
				}
			}
		}
		for(IndexedWord[] pa : pairs){
			allPath.addAll(graph.getAllEdges(pa[0], pa[1]));
		}
		boolean containsSubjectVerb = false;
		for(SemanticGraphEdge edge : allPath){
			GrammaticalRelation gr = edge.getRelation();
			if(gr.toString().equals("nsubj") || gr.toString().equals("ccomp")
					|| gr.toString().equals("nsubjpass") || gr.toString().equals("dobj")){
				IndexedWord gov = edge.getGovernor();
				if(gov.index() < startNode.index() || gov.index() > endNode.index())
					continue;
				containsSubjectVerb = true;
				continue;
			}
			
		}
		
		if(containsSubjectVerb == true)
			return null;
		
		boolean find = false;	
		for(SemanticGraphEdge edge : allPath){
			GrammaticalRelation gr = edge.getRelation();
			if (gr.toString().equals("det") || gr.toString().equals("amod") || gr.toString().equals("poss")
					|| gr.toString().equals("num") || gr.toString().equals("advmod")) {
				IndexedWord gov = edge.getGovernor();
				if(!gov.tag().startsWith("NN")) 
					continue;
				if(gov.index() < startNode.index() || gov.index() > endNode.index())
					continue;
				Argument head = new Argument();
				if(getIndex(arg, gov) == -1){
					continue;
				}
				head.add(arg.get(getIndex(arg, gov)));
				find  = true;
				return head;	
			}else if(gr.toString().equals("pobj") || gr.toString().equals("tmod") || gr.toString().equals("nn")){
				IndexedWord dep = edge.getDependent();
				if(!dep.tag().startsWith("NN")) 
					continue;
				if(dep.index() < startNode.index() || dep.index() > endNode.index())
					continue;
				Argument head = new Argument();
				if(getIndex(arg, dep) == -1){
					continue;
				}
				head.add(arg.get(getIndex(arg, dep)));
				find  = true;
				return head;
				
			}
		}
		
		if(find == false){
			for(int i=0; i<arg.size(); i++){
				if(arg.get(i).tag().equals("NNS")){
					Argument head = new Argument();
					head.add(arg.get(i));
					return head;
					
				}
			}
			
		}
		
		return null;
	}
		
	public static void dfs(Tree node, Tree parent, HeadFinder headFinder,
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
