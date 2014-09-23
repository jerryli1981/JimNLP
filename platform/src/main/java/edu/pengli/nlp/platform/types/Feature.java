package edu.pengli.nlp.platform.types;

public class Feature {
	
	private String Name;
	private String Value;
	
	public Feature(String featureName, String featureValue){
		Name = featureName;
		Value =featureValue;
	}

	public int hashCode(){
		
		return Name.hashCode()+Value.hashCode();
		
	}
	public boolean equals(Object compare) {

		if (compare instanceof Feature) {
			Feature obj = (Feature) compare;
			if (this.Name.equals(obj.Name) && this.Value.equals(obj.Value))
				return true;
		}
		return false;
	}
	public String getValue(){
		return Value;
	}
	public String toString(){
	    return Name+":"+Value  ;
	}
	
}
