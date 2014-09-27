package edu.pengli.nlp.platform.types;


public class Summary extends InstanceList {
	

	public Summary(){
		super(null);
	}
	  
	public int length(){
		int length = 0;
		for(Instance sent : this){
			length += ((String)sent.getSource()).split(" ").length;
		}
		return length;
	}

}
