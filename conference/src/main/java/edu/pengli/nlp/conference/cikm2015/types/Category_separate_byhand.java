package edu.pengli.nlp.conference.cikm2015.types;

import java.util.LinkedHashMap;
import java.util.Map;

public enum Category_separate_byhand {

	Accidents_and_Natural_Disasters(1, generateAspectsFor1()), 
	Attacks(2,generateAspectsFor2()), 
	Health_and_Safety(3, generateAspectsFor3()), 
	Endangered_Resources(4, generateAspectsFor4()), 
	Investigations_and_Trials(5,generateAspectsFor5());

	private int idx;
	private static Map<String, String[]> type1aspects;
	private static Map<String, String[]> type2aspects;
	private static Map<String, String[]> type3aspects;
	private static Map<String, String[]> type4aspects;
	private static Map<String, String[]> type5aspects;

	public int getId() {
		return idx;
	}

	public static Map<String, String[]> getAspects(int id) {
		switch (id) {
		case 1:
			return generateAspectsFor1();
		case 2:
			return generateAspectsFor2();

		case 3:
			return generateAspectsFor3();
		case 4:
			return generateAspectsFor4();
		case 5:
			return generateAspectsFor5();
			
		default: return null;
		}
	}

	private static Map<String, String[]> generateAspectsFor1() {
		type1aspects = new LinkedHashMap<String, String[]>();
/*		String[] keywordsForWhat = {"calamity", "casualty", "collision", 
				"crash", "wreck", "incident", "pile", "smash", "catastrophe", "act god",
				"explosion", "cave", "collapse", "train", "airplane", "ship", "boat", "submarine", 
				"automobile", "auto", "car", "bus", "truck", "oil spill", "fire", "pedestrian", 
				"bicycle", "cyclist", "nuclear", "earthquake", "eruption", "flash", "flood", "hurricane", 
				"landslide", "mudslide", "slide", "tornado", "tsunami", "volcano", "wildfire", "wild fire", 
				"blizzard", "snow", "ice", "storm"};
		type1aspects.put("what", keywordsForWhat);*/
		
		String[] keywordsForWhat1 = {"calamity", "casualty", "collision", 
				"crash", "wreck", "incident"};
		type1aspects.put("what1", keywordsForWhat1);
		
		String[] keywordsForWhat2 = {"smash", "catastrophe", "act god",
				"explosion", "cave", "collapse"};
		type1aspects.put("what2", keywordsForWhat2);
		
		String[] keywordsForWhat3 = {"train", "airplane", "ship", "boat", "submarine", 
				"automobile", "auto", "car", "bus", "truck"};
		type1aspects.put("what3", keywordsForWhat3);
		
		String[] keywordsForWhat4 = {"oil spill", "fire", "pedestrian", 
				"bicycle", "cyclist", "nuclear", "earthquake", "eruption", "flash", "flood", "hurricane", 
				"landslide", "mudslide"};
		type1aspects.put("what4", keywordsForWhat4);
		
		String[] keywordsForWhat5 = {"slide", "tornado", "tsunami", "volcano", "wildfire", "wild fire", 
				"blizzard", "snow", "ice", "storm"};
		type1aspects.put("what5", keywordsForWhat5);
		

/*		String[] keywordsForWhen = { "january", "jan", "february", "feb", "march", "mar", "april", "apr", "may", 
				"june", "jun", "july", "jul", "august", "aug", "september", "sept", "sep", "october", 
				"oct", "november", "nov", "december", "dec", "monday", "mon", "tuesday", "tues", "tue", 
				"wednesday", "wed", "thursday", "thurs", "thur", "friday", "fri", "saturday", "sat", 
				"sunday", "sun", "yesterday", "today"};
		type1aspects.put("when", keywordsForWhen);*/
		
		String[] keywordsForWhen1 = { "january", "jan", "february", "feb", "march", "mar", "april", "apr", "may", 
				"june", "jun", "july", "jul", "august", "aug", "september", "sept", "sep", "october", 
				"oct", "november", "nov", "december", "dec"};
		type1aspects.put("when1", keywordsForWhen1);
		
		String[] keywordsForWhen2 = { "monday", "mon", "tuesday", "tues", "tue", 
				"wednesday", "wed", "thursday", "thurs", "thur", "friday", "fri", "saturday", "sat", 
				"sunday", "sun"};
		type1aspects.put("when2", keywordsForWhen2);
		
		String[] keywordsForWhen3 = {"yesterday", "today"};
		type1aspects.put("when3", keywordsForWhen3);


/*		String[] keywordsForWhere = {"location", "locality", "point", "position", "site", "spot", "place", 
				"nearby", "close to", "around", "adjacent to", "intersection", "street", "road", "highway",
				"boulevard", "blvd", "interstate", "outside", "inside", "near", "landfall", "hit", "epicenter"};
		type1aspects.put("where", keywordsForWhere);*/
		
		String[] keywordsForWhere1 = {"location", "locality", "point", "position", "site", "spot", "place" 
				};
		type1aspects.put("where1", keywordsForWhere1);
		
		String[] keywordsForWhere2 = {"nearby", "close to", "around", "adjacent to", "intersection", "street", "road", "highway",
				"boulevard", "blvd", "interstate"};
		type1aspects.put("where2", keywordsForWhere2);
		
		String[] keywordsForWhere3 = {"outside", "inside", "near",  "landfall", "hit", "epicenter"};
		type1aspects.put("where3", keywordsForWhere3);

/*		String[] keywordsForWhy = {"broken", "faulty", "careless", "inattention", "bad judgment", "poor judgment", 
				"fatigue", "faulty equipment", "defective equipment", "equipment failure", "weather related", "weather", 
				"design", "cause", "explanation", "explain", "reason", "precipitate", "distract", "drug", "alcohol", 
				"due to", "thermonuclear activity", "seismic activity", "seismic", "plate movement", "tectonic movement", 
				"plate", "tectonic", "movement", "human exploration", "human involvement", "human inteference", "due to"};
		type1aspects.put("why", keywordsForWhy);*/
		
		String[] keywordsForWhy1 = {"broken", "faulty", "careless", "inattention", "bad judgment", "poor judgment", 
				"fatigue"};
		type1aspects.put("why1", keywordsForWhy1);
		
		String[] keywordsForWhy2 = {"faulty equipment", "defective equipment", "equipment failure", "weather related", "weather", 
				"design", "cause"};
		type1aspects.put("why2", keywordsForWhy2);
		
		String[] keywordsForWhy3 = {"explanation", "explain", "reason", "precipitate", "distract", "drug", "alcohol", 
				"due to", "thermonuclear activity", "seismic activity"};
		type1aspects.put("why3", keywordsForWhy3);
		
		
		String[] keywordsForWhy4 = {"seismic", "plate movement", "tectonic movement", 
				"plate", "tectonic", "movement", "human exploration", "human involvement", "human inteference", "due to"};
		type1aspects.put("why4", keywordsForWhy4);
		

/*		String[] keywordsForWhoaffected = {"casualty",  "dead", "wound", "injur", "loss of life", "decease", 
				"demise", "mortality", "hurt", "harm", "survivor", "casualt", "death toll", "death", "dead", "wound", "injur", "loss of life", 
				"decease", "demise", "mortality", "hurt", "harm", "death toll", "death", "missing", "threat", "survivor"};
		type1aspects.put("whoaffected", keywordsForWhoaffected);*/
		
		String[] keywordsForWhoaffected1 = {"casualty",  "dead", "wound", "injur", "loss of life", "decease"};
		type1aspects.put("whoaffected1", keywordsForWhoaffected1);
		
		String[] keywordsForWhoaffected2 = {"demise", "mortality", "hurt", "harm", "survivor", "casualt", "death toll", "death", "dead", "wound", "injur"};
		type1aspects.put("whoaffected2", keywordsForWhoaffected2);
		
		String[] keywordsForWhoaffected3 = {"loss of life", 
				"decease", "demise", "mortality", "hurt", "harm", "death toll", "death", "missing", "threat", "survivor"};
		type1aspects.put("whoaffected3", keywordsForWhoaffected3);

/*		String[] keywordsForDamages = { "property", "possession", "belongings", "goods", "assets", "fiscal", 
				"money", "economic", "financial", "loss", "damage", "ruin", "burn", "property loss", "financial loss", "property", "possession", 
				"belongings", "goods", "assets", "fiscal", "money", "economic", "property loss", 
				"financial loss","financial", "loss", "damage to", "disruption of", "damage", "ruin", "burn"};
		type1aspects.put("damages", keywordsForDamages);*/
		
		String[] keywordsForDamages1 = { "property", "possession", "belongings", "goods", "assets", "fiscal", 
				"money", "economic", "financial"};
		type1aspects.put("damages1", keywordsForDamages1);
		
		String[] keywordsForDamages2 = {"loss", "damage", "ruin", "burn", "property loss", "financial loss", "property", "possession", 
				"belongings", "goods", "assets", "fiscal", "money", "economic"};
		type1aspects.put("damages2", keywordsForDamages2);
		
		String[] keywordsForDamages3 = {"property loss", 
				"financial loss","financial", "loss", "damage to", "disruption of", "damage", "ruin", "burn"};
		type1aspects.put("damages3", keywordsForDamages3);

/*		String[] keywordsForCountermeasures = {"safety", "prevent", "rescue", "hazardous condition", "unsafe condition", "hazardous", "unsafe", "maintenance", "repair", 
				"correct", "recovery plan", "prevent", "prevention effort", "rescue effort",  "relief effort", "relief", "emergency plan", 
				"risk reduction", "preparedness"};
		type1aspects.put("countermeasures", keywordsForCountermeasures);*/
		
		String[] keywordsForCountermeasures1 = {"safety", "prevent", "rescue", "hazardous condition", "unsafe condition", "hazardous", "unsafe", "maintenance", "repair", 
				"correct", "recovery plan", "prevent", "prevention effort"};
		type1aspects.put("countermeasures1", keywordsForCountermeasures1);
		
		String[] keywordsForCountermeasures2 = {"rescue effort",  "relief effort", "relief", "emergency plan", 
				"risk reduction", "preparedness"};
		type1aspects.put("countermeasures2", keywordsForCountermeasures2);
		
		
		return type1aspects;
	}

	private static Map<String, String[]> generateAspectsFor2() {
		type2aspects = new LinkedHashMap<String, String[]>();
/*		String[] keywordsForWhat = { "felony", "lawlessness", "malfeasance", "misdemeanor", "transgression", "unlawful", "violation", 
				"wrongdoing", "raid", "ambush", "assault", "insurrection", "violence", "calculated", "threat", 
				 "riot", "violence", "raid", "ambush", "assault", "revolt", "violence", "massacre", "slaughter",
				 "assault", "battery", "burglary", "theft", "home invasion", "murder", "rape", "child molestation", "molestation", 
				 "sexual abuse", "abuse", "robbery", "manslaughter", "kidnapping", "abduction", "poisoning", "shooting", 
				 "endangerment", "hate crime", "political", "religious", "guerilla", "nuclear", "chemical weapon", "biological weapon", 
				 "chemic", "biologic", "weapon", "weapon of mass destruction", "bomb", "mass destruction",
				 "bomb", "gunfire", "landmine", "explosion", "nuclear", "uprising"};
		type2aspects.put("what", keywordsForWhat);*/
		
		String[] keywordsForWhat = { "felony", "lawlessness", "malfeasance", "misdemeanor", "transgression", "unlawful", "violation", 
				"wrongdoing", "raid", "ambush", "assault", "insurrection", "violence", "calculated", "threat", 
				 "riot", "violence", "raid", "ambush", "assault", "revolt", "violence", "massacre", "slaughter",
				 "assault", "battery", "burglary", "theft", "home invasion", "murder", "rape", "child molestation", "molestation", 
				 "sexual abuse", "abuse", "robbery", "manslaughter", "kidnapping", "abduction", "poisoning", "shooting", 
				 "endangerment", "hate crime", "political", "religious", "guerilla", "nuclear", "chemical weapon", "biological weapon", 
				 "chemic", "biologic", "weapon", "weapon of mass destruction", "bomb", "mass destruction",
				 "bomb", "gunfire", "landmine", "explosion", "nuclear", "uprising"};
		type2aspects.put("what", keywordsForWhat);
		
		String[] keywordsForWhat1 = { "felony", "lawlessness", "malfeasance", "misdemeanor", "transgression", "unlawful", "violation", 
				"wrongdoing", "raid", "ambush", "assault", "insurrection", "violence", "calculated", "threat", 
				 "riot", "violence", "raid", "ambush", "assault"};
		type2aspects.put("what1", keywordsForWhat1);
		
		String[] keywordsForWhat2 = {"revolt", "violence", "massacre", "slaughter",
				 "assault", "battery", "burglary", "theft", "home invasion", "murder", "rape", "child molestation", "molestation", 
				 "sexual abuse", "abuse", "robbery", "manslaughter", "kidnapping", "abduction", "poisoning", "shooting", 
				 "endangerment", "hate crime"};
		type2aspects.put("what2", keywordsForWhat2);
		
		String[] keywordsForWhat3 = {"political", "religious", "guerilla", "nuclear", "chemical weapon", "biological weapon", 
				 "chemic", "biologic", "weapon", "weapon of mass destruction", "bomb", "mass destruction",
				 "bomb", "gunfire", "landmine", "explosion", "nuclear", "uprising"};
		type2aspects.put("what3", keywordsForWhat3);
		

/*		String[] keywordsForWhen = {"january", "jan", "february", "feb", "march", "mar", "april", "apr", "may", "june", 
				"jun", "july", "jul", "august", "aug", "september", "sept", "sep", "october", "oct", "november", "nov", 
				"december", "dec", "monday", "mon", "tuesday", "tues", "tue", "wednesday", "wed", "thursday", "thurs", 
				"thur", "friday", "fri", "saturday", "sat", "sunday", "sun", "yesterday", "today"};
		type2aspects.put("when", keywordsForWhen);

		String[] keywordsForWhere = { "location", "locality", "point", "position", "site", "spot", "place", "nearby", 
				"close to", "around", "adjacent to", "street", "road", "highway", "boulevard", "blvd", "interstate", 
				"outside", "inside", "near"};
		type2aspects.put("where", keywordsForWhere);*/
		
		String[] keywordsForWhere1 = {"location", "locality", "point", "position", "site", "spot", "place" 
		};
type2aspects.put("where1", keywordsForWhere1);

String[] keywordsForWhere2 = {"nearby", "close to", "around", "adjacent to", "intersection", "street", "road", "highway",
		"boulevard", "blvd", "interstate"};
type2aspects.put("where2", keywordsForWhere2);

String[] keywordsForWhere3 = {"outside", "inside", "near",  "landfall", "hit", "epicenter"};
type2aspects.put("where3", keywordsForWhere3);

String[] keywordsForWhen1 = { "january", "jan", "february", "feb", "march", "mar", "april", "apr", "may", 
		"june", "jun", "july", "jul", "august", "aug", "september", "sept", "sep", "october", 
		"oct", "november", "nov", "december", "dec"};
type2aspects.put("when1", keywordsForWhen1);

String[] keywordsForWhen2 = { "monday", "mon", "tuesday", "tues", "tue", 
		"wednesday", "wed", "thursday", "thurs", "thur", "friday", "fri", "saturday", "sat", 
		"sunday", "sun"};
type2aspects.put("when2", keywordsForWhen2);

String[] keywordsForWhen3 = {"yesterday", "today"};
type2aspects.put("when3", keywordsForWhen3);



/*		String[] keywordsForWhy = { "motive", "reason", "commit", "intent", "mens rea", "cause", "means", "opportunity",
				"politic", "religio", "ideologic", "revenge", "retaliation", "disrupt", "disturb", "instability", "unstable",
				};
		type2aspects.put("why", keywordsForWhy);*/

String[] keywordsForWhy1 = { "motive", "reason", "commit", "intent", "mens rea", "cause", "means", "opportunity",
		"politic", "religio", "ideologic"};
type2aspects.put("why1", keywordsForWhy1);

String[] keywordsForWhy2 = {"revenge", "retaliation", "disrupt", "disturb", "instability", "unstable",
		};
type2aspects.put("why2", keywordsForWhy2);

/*		String[] keywordsForWhoaffected = { "casualty", "death", "injur", "dead", "loss", "wound", "hurt", "hostage",
				"casualt", "death", "loss", "injur", "damage", "dead", "missing", "wound", "harm", "captive", "hostage",
				"casualt", "death", "injur", "loss", "dead", "wound", "missing", "captive", "hostage", "harm"};
		type2aspects.put("whoaffected", keywordsForWhoaffected);*/

String[] keywordsForWhoaffected1 = { "casualty", "death", "injur", "dead", "loss", "wound", "hurt", "hostage",
		"casualt", "death", "loss", "injur"};
type2aspects.put("whoaffected1", keywordsForWhoaffected1);

String[] keywordsForWhoaffected2 = {"damage", "dead", "missing", "wound", "harm", "captive", "hostage",
		"casualt", "death", "injur", "loss", "dead", "wound", "missing", "captive", "hostage", "harm"};
type2aspects.put("whoaffected2", keywordsForWhoaffected2);

/*		String[] keywordsForDamages = {"property", "possession", "belongings", "goods", 
				"assets", "fiscal", "money", "economic", "financial", "loss", "damage", "ruin", "burn", 
				"smash", "destroy", "arson", "property loss", "financial loss", "property", "possession", "belongings", "goods", 
				"assets", "fiscal", "money", "economic", "financial", "loss", "damage to", "disruption of", "damage", 
				"ruin", "burn", "damage to", "disruption of", "property loss", "financial loss", "property", "possession", "belongings", 
				"goods", "assets", "fiscal", "money", "economic", "financial", "loss", "damage to", "disruption of", 
				"damage", "ruin", "burn", "damage to", "disruption of",  "property loss", "financial loss"};
		type2aspects.put("damages", keywordsForDamages);*/
		
		String[] keywordsForDamages1 = { "property", "possession", "belongings", "goods", "assets", "fiscal", 
				"money", "economic", "financial"};
		type2aspects.put("damages1", keywordsForDamages1);
		
		String[] keywordsForDamages2 = {"loss", "damage", "ruin", "burn", "property loss", "financial loss", "property", "possession", 
				"belongings", "goods", "assets", "fiscal", "money", "economic"};
		type2aspects.put("damages2", keywordsForDamages2);
		
		String[] keywordsForDamages3 = {"property loss", 
				"financial loss","financial", "loss", "damage to", "disruption of", "damage", "ruin", "burn"};
		type2aspects.put("damages3", keywordsForDamages3);

/*		String[] keywordsForCountermeasures = {"secur",  "prevent", "police", 
				"light", "surveillance camera", "surveillan", "camera", "patrol", "aware", 
				"aware", "suspicious behavior", "suspicious", "unusual", "out of the ordinary", "watch list",
				"secur", "prevent", "patrol", "aware", "surveillance", "police response","risk analysis", "police", "suspicious", "unusual"};
		type2aspects.put("countermeasures", keywordsForCountermeasures);*/
		
		String[] keywordsForCountermeasures1 = {"secur",  "prevent", "police", 
				"light", "surveillance camera", "surveillan", "camera", "patrol", "aware", 
				"aware", "suspicious behavior", "suspicious"};
		type2aspects.put("countermeasures1", keywordsForCountermeasures1);
		
		String[] keywordsForCountermeasures2 = {"unusual", "out of the ordinary", "watch list",
				"secur", "prevent", "patrol", "aware", "surveillance", "police response","risk analysis", "police", "suspicious", "unusual"};
		type2aspects.put("countermeasures2", keywordsForCountermeasures2);

/*		String[] keywordsForPerpetrators = {"perpetrate", "commit", "culprit", "criminal", "incendiar", "wrongdoer", 
				"guilty party", "agent", "offender", "perp", "thug", "vandal", "accused", "suspect",
				"terrorist", "subversive", "bomber", "guerilla", "incendiar", "radical", "rebel", "revolutionary", 
				"jihadist", "anarchist", "insurgent", "attacker", "aggressor", "assailant", "assaulter", "mugger"};
		type2aspects.put("perpetrators", keywordsForPerpetrators);*/
		
		String[] keywordsForPerpetrators1 = {"perpetrate", "commit", "culprit", "criminal", "incendiar", "wrongdoer", 
				"guilty party", "agent", "offender", "perp"};
		type2aspects.put("perpetrators1", keywordsForPerpetrators1);
		
		String[] keywordsForPerpetrators2 = {"thug", "vandal", "accused", "suspect",
				"terrorist", "subversive", "bomber", "guerilla"};
		type2aspects.put("perpetrators2", keywordsForPerpetrators2);
		
		String[] keywordsForPerpetrators3 = {"incendiar", "radical", "rebel", "revolutionary", 
				"jihadist", "anarchist", "insurgent", "attacker", "aggressor", "assailant", "assaulter", "mugger"};
		type2aspects.put("perpetrators3", keywordsForPerpetrators3);
		
		
		return type2aspects;
	}

	private static Map<String, String[]> generateAspectsFor3() {
		type3aspects = new LinkedHashMap<String, String[]>();
/*		String[] keywordsForWhat = {"fitness", "physical", "condition", "wellness", "wellbeing", "welfare", 
				"shape", "disease", "disorder", "guard", "injury", "prevent", "accident", "guard", 
				"protect", "precaution", "safeguard", "prevention", "public", "child", "animal", 
				"community", "senior", "geront", "mental", "accident", "attack", "disaster"};
		type3aspects.put("what", keywordsForWhat);*/
		
		String[] keywordsForWhat1 = {"fitness", "physical", "condition", "wellness", "wellbeing", "welfare", 
				"shape", "disease"};
		type3aspects.put("what1", keywordsForWhat1);
		
		String[] keywordsForWhat2 = {"disorder", "guard", "injury", "prevent", "accident", "guard", 
				"protect", "precaution", "safeguard"};
		type3aspects.put("what2", keywordsForWhat2);
		
		String[] keywordsForWhat3 = {"prevention", "public", "child", "animal", 
				"community", "senior", "geront", "mental", "accident", "attack", "disaster"};
		type3aspects.put("what3", keywordsForWhat3);

/*		String[] keywordsForWhy = {"malnutrition", "disease", "poverty", "infectious disease", "preventable disease", "cardiovascular disease", 
				 "obesity", "medical care", "medical access", "lack of sanitation", 
				"unsanitary", "due to", "attack", "accident", "disaster", "careless", "fatigue", "shortcuts", "due to"};
		type3aspects.put("why", keywordsForWhy);*/
		
		String[] keywordsForWhy1 = {"malnutrition", "disease", "poverty", "infectious disease", "preventable disease", "cardiovascular disease"};
		type3aspects.put("why1", keywordsForWhy1);
		
		String[] keywordsForWhy2 = { "obesity", "medical care", "medical access", "lack of sanitation", 
				"unsanitary"};
		type3aspects.put("why2", keywordsForWhy2);
		
		String[] keywordsForWhy3 = {"attack", "accident", "disaster", "careless", "fatigue", "shortcuts", "due to"};
		type3aspects.put("why3", keywordsForWhy3);

/*		String[] keywordsForWhoaffected = {"dead", "ill", "family", "relative", "community", "population", "elder", 
				"young", "children", "infants", "aged", "compromised", "injure", "survivor", "community"};
		type3aspects.put("whoaffected", keywordsForWhoaffected);*/
		
		String[] keywordsForWhoaffected1 = {"dead", "ill", "family", "relative", "community"};
		type3aspects.put("whoaffected1", keywordsForWhoaffected1);
		
		String[] keywordsForWhoaffected2 = {"population", "elder", 
				"young", "children", "infants"};
		type3aspects.put("whoaffected2", keywordsForWhoaffected2);
		
		String[] keywordsForWhoaffected3 = {"aged", "compromised", "injure", "survivor", "community"};
		type3aspects.put("whoaffected3", keywordsForWhoaffected3);

/*		String[] keywordsForHow = {"illness", "injur", "sick", "incapacit", "invalid", "death", "malnourish", "obesity", 
				"loss of life", "coma", "unconscious", "mortalit",  "injur", "death", "dismemberment", "loss of life", "loss of limb"};
		type3aspects.put("how", keywordsForHow);*/
		
		String[] keywordsForHow1 = {"illness", "injur", "sick", "incapacit", "invalid", "death"};
		type3aspects.put("how1", keywordsForHow1);
		
		String[] keywordsForHow2 = {"malnourish", "obesity", 
				"loss of life", "coma", "unconscious"};
		type3aspects.put("how2", keywordsForHow2);
		
		String[] keywordsForHow3 = {"mortalit",  "injur", "death", "dismemberment", "loss of life", "loss of limb"};
		type3aspects.put("how3", keywordsForHow3);

/*		String[] keywordsForCountermeasures = {"educate", "prevent", "maintenance", "universal health care", "primary health care", "health care", "immuniz", "medic", 
				 "repair", "educat"};
		type3aspects.put("countermeasures", keywordsForCountermeasures);*/
		
		String[] keywordsForCountermeasures1 = {"primary health care", "health care", "immuniz", "medic", 
				 "repair", "educat"};
		type3aspects.put("countermeasures1", keywordsForCountermeasures1);
		
		String[] keywordsForCountermeasures2 = {"educate", "prevent", "maintenance", "universal health care"};
		type3aspects.put("countermeasures2", keywordsForCountermeasures2);
		
		
		return type3aspects;
	}

	private static Map<String, String[]> generateAspectsFor4() {
		type4aspects = new LinkedHashMap<String, String[]>();
		
		
/*		String[] keywordsForWhat = {"destruction", "dying", "vulnerable", "extinction", "depletion", "lessen", "disappear", 
				"decrease", "threat", "conservation", "environment", "forest", "mineral", "oil", "gas", "coal", "marsh", "wetland", 
				"reef", "river", "land", "water supply", "waterway", "water", "native plant", "native", "endangered speci", "endangered animal", 
				"endangered plant", "endanger", "plant", "animal kingdom", "animal", "fish", "wildlife", 
				"wildflowers", "habitat", "culture", "biodiversity"};
		type4aspects.put("what", keywordsForWhat);*/
		
		String[] keywordsForWhat1 = {"destruction", "dying", "vulnerable", "extinction", "depletion", "lessen", "disappear", 
				"decrease", "threat", "conservation"};
		type4aspects.put("what1", keywordsForWhat1);
		
		String[] keywordsForWhat2 = {"environment", "forest", "mineral", "oil", "gas", "coal", "marsh", "wetland", 
				"reef", "river", "land", "water supply", "waterway", "water", "native plant", "native", "endangered speci", "endangered animal", 
				"endangered plant", "endanger"};
		type4aspects.put("what2", keywordsForWhat2);
		
		String[] keywordsForWhat3 = {"plant", "animal kingdom", "animal", "fish", "wildlife", 
				"wildflowers", "habitat", "culture", "biodiversity"};
		type4aspects.put("what3", keywordsForWhat3);

/*		String[] keywordsForImportance = {"sustain", "cure", "economic", "medical discover", "economic development",  "health", "disaster resilience"};
		type4aspects.put("importance", keywordsForImportance);*/
		
		String[] keywordsForImportance1 = {"sustain", "cure"};
		type4aspects.put("importance1", keywordsForImportance1);
		
		String[] keywordsForImportance2 = {"economic", "economic development",};
		type4aspects.put("importance2", keywordsForImportance2);
		
		String[] keywordsForImportance3 = {"medical discover",  "health", "disaster resilience"};
		type4aspects.put("importance3", keywordsForImportance3);

/*		String[] keywordsForThreats = {"disaster", "weather", "pollution", "nonnative speci", "nonnative", "invasive speci", "invasive", "market demand", "urban expansion", 
				"expansion", "natural disaster",  "global warming", "climate change", "war", "conflict", 
				"deforestation", "acid rain", "greenhouse gas", "greenhouse effect", "habitat destruct"};
		type4aspects.put("threats", keywordsForThreats);*/
		
		String[] keywordsForThreats1 = {"disaster", "weather", "pollution", "nonnative speci", "nonnative", "invasive speci", "invasive", "market demand", "urban expansion", 
				};
		type4aspects.put("threats1", keywordsForThreats1);
		
		String[] keywordsForThreats2 = {
				"expansion", "natural disaster",  "global warming", "climate change", "war", "conflict"};
		type4aspects.put("threats2", keywordsForThreats2);
		
		String[] keywordsForThreats3 = {
				"deforestation", "acid rain", "greenhouse gas", "greenhouse effect", "habitat destruct"};
		type4aspects.put("threats3", keywordsForThreats3);

/*		String[] keywordsForCountermeasures = {"cloning", "conservation", "preservation", "wildlife refuge", "recycling", "environmental protection", 
				"habitat protection", "environment", "protect", "behavior", "renewable energy", "activism", "educat", "breeding program", "zoo", "habitat"};
		type4aspects.put("countermeasures", keywordsForCountermeasures);*/
		
		String[] keywordsForCountermeasures1 = {"cloning", "conservation", "preservation"};
		type4aspects.put("countermeasures1", keywordsForCountermeasures1);
		
		String[] keywordsForCountermeasures2 = {"wildlife refuge", "recycling", "environmental protection", 
				"habitat protection", "environment", "protect", "behavior"};
		type4aspects.put("countermeasures2", keywordsForCountermeasures2);
		
		String[] keywordsForCountermeasures3 = {"renewable energy", "activism", "educat", "breeding program", "zoo", "habitat"};
		type4aspects.put("countermeasures3", keywordsForCountermeasures3);
		
		
		
		return type4aspects;
	}

	private static Map<String, String[]> generateAspectsFor5() {
		type5aspects = new LinkedHashMap<String, String[]>();
		
		
/*		String[] keywordsForWhat = {"inquiry", "enquiry", "inquest", "probe", "check",
				"hearing", "legal proceeding", "tribunal", "suit", "court", "case",
				"criminal", "political", "civil", "negligence", "corporat", "background", 
				"personal", "security", "forensic", "criminal", "civil", "war crimes", "war"};
		type5aspects.put("what", keywordsForWhat);*/
		
		String[] keywordsForWhat1 = {"inquiry", "enquiry", "inquest", "probe", "check",
				"hearing"};
		type5aspects.put("what1", keywordsForWhat1);
		
		String[] keywordsForWhat2 = { "legal proceeding", "tribunal", "suit", "court", "case",
				"criminal", "political", "civil"};
		type5aspects.put("what2", keywordsForWhat2);
		
		String[] keywordsForWhat3 = {"negligence", "corporat", "background", 
				"personal", "security", "forensic", "criminal", "civil", "war crimes", "war"};
		type5aspects.put("what3", keywordsForWhat3);
		

/*		String[] keywordsForWho = {"suspect", "accuse", "culprit", "offend", "violat", 
				"defendant", "plaintiff", "convict", "defense", "appellant"};
		type5aspects.put("who", keywordsForWho);*/
		
		String[] keywordsForWho1 = {"suspect", "accuse", "culprit", "offend", "violat", 
				"defendant"};
		type5aspects.put("who1", keywordsForWho1);
		
		String[] keywordsForWho2 = {"plaintiff", "convict", "defense", "appellant"};
		type5aspects.put("who2", keywordsForWho2);

/*		String[] keywordsForWho_Inv = {"auditor", "commissioner", "constable", "county clerk", 
				"deputy", "inspectory", "marshal", "police officer", "police", "officer", "private detective", 
				"private investigator", "detective", "investigat", "sheriff", "undercover", "magistrate", 
				"coroner", "controller", "comptroller", "judge", "constable", "magistrate", "tribunal", 
				"arbitrator", "adjudicator", "judiciary", "court"};
		type5aspects.put("who_inv", keywordsForWho_Inv);*/
		
		String[] keywordsForWho_Inv1 = {"auditor", "commissioner", "constable", "county clerk", 
				"deputy", "inspectory", "marshal"};
		type5aspects.put("who_inv1", keywordsForWho_Inv1);
		
		String[] keywordsForWho_Inv2 = {"police officer", "police", "officer", "private detective", 
				"private investigator", "detective", "investigat", "sheriff", "undercover", "magistrate", 
				"coroner", "controller", "comptroller"};
		type5aspects.put("who_inv2", keywordsForWho_Inv2);
		
		String[] keywordsForWho_Inv3 = {"judge", "constable", "magistrate", "tribunal", 
				"arbitrator", "adjudicator", "judiciary", "court"};
		type5aspects.put("who_inv3", keywordsForWho_Inv3);

/*		String[] keywordsForWhy = {"accident",  "death", "crime", "corruption", "theft", "fire", "abuse", "due to",
				"determine guilt", "determine innocen", "guilt", "innocen", "wrongdoing", "charge", "justice", "verdict"};
		type5aspects.put("why", keywordsForWhy);*/
		
		String[] keywordsForWhy1 = {"accident",  "death", "crime", "corruption", "theft", "fire", "abuse", "due to",
				};
		type5aspects.put("why1", keywordsForWhy1);
		
		String[] keywordsForWhy2 = {
				"determine guilt", "determine innocen", "guilt", "innocen"};
		type5aspects.put("why2", keywordsForWhy2);
		
		String[] keywordsForWhy3 = {"wrongdoing", "charge", "justice", "verdict"};
		type5aspects.put("why3", keywordsForWhy3);

/*		String[] keywordsForCharges = {"accus", "blame", "wrongdoing", "crime", "conspiracy", "violence", "drug", "alcohol", "offense", 
				"victim", "steal", "theft", "weapon", "crime", "felony", "misdemeanor", "arson", "assault", "burglary", "murder", 
				"rape", "abuse", "robbery"};
		type5aspects.put("charges", keywordsForCharges);*/
		
		String[] keywordsForCharges1 = {"accus", "blame", "wrongdoing", "crime", "conspiracy", "violence", "drug", "alcohol", "offense", 
				};
		type5aspects.put("charges1", keywordsForCharges1);
		
		String[] keywordsForCharges2 = {
				"victim", "steal", "theft", "weapon", "crime", "felony", "misdemeanor", "arson"};
		type5aspects.put("charges2", keywordsForCharges2);
		
		String[] keywordsForCharges3 = {"assault", "burglary", "murder", 
				"rape", "abuse", "robbery"};
		type5aspects.put("charges3", keywordsForCharges3);

/*		String[] keywordsForPlead = {"innocent", "denial", "admission", "wrongdoing", "circumstance", "explain", 
				"explanation", "charges", "guilty", "not guilty", "no contest", "insanity", "nolo contendere", "failure to appear", 
				"refus", "plea bargain", "drop charges", "reduce charges", "reduction of charges", "drop", "reduc", 
				"charges"};
		type5aspects.put("plead", keywordsForPlead);*/
		
		String[] keywordsForPlead1 = {"innocent", "denial", "admission", "wrongdoing", "circumstance", "explain"
				};
		type5aspects.put("plead1", keywordsForPlead1);
		
		String[] keywordsForPlead2 = { 
				"explanation", "charges", "guilty", "not guilty", "no contest", "insanity", "nolo contendere", "failure to appear", 
				};
		type5aspects.put("plead2", keywordsForPlead2);
		
		String[] keywordsForPlead3 = {"refus", "plea bargain", "drop charges", "reduce charges", "reduction of charges", "drop", "reduc", 
				"charges"};
		type5aspects.put("plead3", keywordsForPlead3);

/*		String[] keywordsForSentence = {"admonishment", "fine", "charge", "reprimand", "counsel", "recommend", "forfeiture",
				"community service", "life sentence", "death sentence", "life", "death", "sentence", "fine", "forfeiture", 
				"imprison", "incarcerat", "jail time", "month", "year", "probation", "treatment", "monitor", 
				"counsel", "reprimand"};
		type5aspects.put("sentence", keywordsForSentence);*/
		
		String[] keywordsForSentence1 = {"admonishment", "fine", "charge", "reprimand", "counsel", "recommend", "forfeiture",
				};
		type5aspects.put("sentence1", keywordsForSentence1);
		
		String[] keywordsForSentence2 = {"community service", "life sentence", "death sentence", "life", "death", "sentence", "fine", "forfeiture", 
				"imprison", "incarcerat"};
		type5aspects.put("sentence2", keywordsForSentence2);
		
		String[] keywordsForSentence3 = {"jail time", "month", "year", "probation", "treatment", "monitor", 
				"counsel", "reprimand"};
		type5aspects.put("sentence3", keywordsForSentence3);
		
		
		return type5aspects;
	}

	private Category_separate_byhand(int idx, Map<String, String[]> aspects) {
		this.idx = idx;
	}

	private Map<String, String[]> getType1Aspects() {
		return type1aspects;
	}

	private Map<String, String[]> getType2Aspects() {
		return type2aspects;
	}

	private Map<String, String[]> getType3Aspects() {
		return type3aspects;
	}

	private Map<String, String[]> getType4Aspects() {
		return type4aspects;
	}

	private Map<String, String[]> getType5Aspects() {
		return type5aspects;
	}
}
