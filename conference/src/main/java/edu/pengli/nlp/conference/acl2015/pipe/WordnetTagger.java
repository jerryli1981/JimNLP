package edu.pengli.nlp.conference.acl2015.pipe;

import java.util.ArrayList;

import org.json.JSONArray;
import org.json.JSONObject;

import edu.pengli.nlp.conference.acl2015.types.Argument;
import edu.pengli.nlp.conference.acl2015.types.Tuple;
import edu.smu.tspell.wordnet.NounSynset;
import edu.smu.tspell.wordnet.Synset;
import edu.smu.tspell.wordnet.SynsetType;
import edu.smu.tspell.wordnet.VerbSynset;
import edu.smu.tspell.wordnet.WordNetDatabase;

public class WordnetTagger {
	
	static WordNetDatabase database;
	static NounSynset nounSynset;
	static NounSynset[] nounHyponyms;
	static VerbSynset verbSynset;
	static VerbSynset[] verbHyponyms;
	
	public WordnetTagger(){
		System.setProperty("wordnet.database.dir", "../models/WordNet/WordNet-3.0/dict");
		database = WordNetDatabase.getFileInstance();
	}
	
	public void annotatePerson(Argument arg1, Argument arg2) throws Exception{
		
		if(arg1.getHead().ner().equals("O")){
			
			if(arg1.getHead().tag().equals("PRP")){
				arg1.getHead().setNER("PERSON");
			}else if(isPerson(arg1.getHead().originalText())){
				arg1.getHead().setNER("PERSON");
			}
	
		}
		
		if(arg2.getHead().ner().equals("O")){
			
			if(arg2.getHead().tag().equals("PRP")){
				arg2.getHead().setNER("PERSON");
			}else if(isPerson(arg2.getHead().originalText())){
				arg2.getHead().setNER("PERSON");
			}
		}
		
	}
	
	public void annotateNoun(Argument arg1, Argument arg2, Tuple t) throws Exception{
		
		if(arg1.getHead().ner().equals("O")){
			String nounArg1 = arg1.getHead().originalText();
			ArrayList<String> nounTypes = getNounTypes(nounArg1);
			if(nounTypes.size() >= 1 && nounTypes.size() <= 3){
				arg1.getHead().setNER(nounTypes.get(0));
			}
	
		}
		
		if(arg2.getHead().ner().equals("O")){
			String nounArg2 = arg2.getHead().originalText();
			ArrayList<String> nounTypes = getNounTypes(nounArg2);
			if(nounTypes.size() >= 1 && nounTypes.size() <= 3){
				arg2.getHead().setNER(nounTypes.get(0));
			}
		}
		
	}
	
	public boolean isPerson(String noun){
		boolean ret = false;
		Synset[] synsets = database.getSynsets(noun, SynsetType.NOUN);
		for (int i = 0; i < synsets.length; i++) {
		    nounSynset = (NounSynset)(synsets[i]);
		    nounHyponyms = nounSynset.getHypernyms();
		    for(int j=0; j<nounHyponyms.length; j++){
		    	String[] wordForms = nounHyponyms[j].getWordForms();
		    	for(int k=0; k<wordForms.length; k++){
		    		if(wordForms[k].contains("person"))
		    			ret = true;
		    	}	
		    }
		} 
		return ret;
		
	}

	public static ArrayList<String> getNounTypes(String noun){
		
		ArrayList<String> ret = new ArrayList<String>();
		Synset[] synsets = database.getSynsets(noun, SynsetType.NOUN);
		for (int i = 0; i < synsets.length; i++) {
		    nounSynset = (NounSynset)(synsets[i]);
		    nounHyponyms = nounSynset.getHypernyms();
		    for(int j=0; j<nounHyponyms.length; j++){
		    	String[] wordForms = nounHyponyms[j].getWordForms();
		    	for(int k=0; k<wordForms.length; k++){
//		    		System.out.println(noun +" is a kind of "+ wordForms[k]);
		    		ret.add(wordForms[k]);
		    	}	
		    }
		    
		} 
		return ret;
	}
	
	public static ArrayList<String> getVerbTypes(String noun){
		
		ArrayList<String> ret = new ArrayList<String>();
		Synset[] synsets = database.getSynsets(noun, SynsetType.VERB);
		for (int i = 0; i < synsets.length; i++) {
		    verbSynset = (VerbSynset)(synsets[i]);
		    verbHyponyms = verbSynset.getHypernyms();
		    for(int j=0; j<verbHyponyms.length; j++){
		    	String[] wordForms = verbHyponyms[j].getWordForms();
		    	for(int k=0; k<wordForms.length; k++){
		    		System.out.println(noun +" is a kind of "+ wordForms[k]);
		    		ret.add(wordForms[k]);
		    	}	
		    } 
		} 
		return ret;
	}
	
	public static void main(String[] args){
		
		WordnetTagger obj = new WordnetTagger();
		System.out.println(obj.getNounTypes("location"));
	}
}
