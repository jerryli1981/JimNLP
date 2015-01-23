package edu.pengli.nlp.conference.acl2015.types;

import java.util.LinkedHashMap;
import java.util.Map;

public enum Category {
	
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
		
		String[] keywordsForWhat = {"calamity", "casualty", "collision", 
				"crash", "wreck", "incident", "pile", "smash", "catastrophe", "act god",
				"explosion", "cave", "collapse", "train", "airplane", "ship", "boat", "submarine", 
				"automobile", "auto", "car", "bus", "truck", "oil spill", "fire", "pedestrian", 
				"bicycle", "cyclist", "nuclear", "earthquake", "eruption", "flash", "flood", "hurricane", 
				"landslide", "mudslide", "slide", "tornado", "tsunami", "volcano", "wildfire", "wild fire", 
				"blizzard", "snow", "ice", "storm"};
		type1aspects.put("what", keywordsForWhat);
		

		String[] keywordsForWhen = { "january", "jan", "february", "feb", "march", "mar", "april", "apr", "may", 
				"june", "jun", "july", "jul", "august", "aug", "september", "sept", "sep", "october", 
				"oct", "november", "nov", "december", "dec", "monday", "mon", "tuesday", "tues", "tue", 
				"wednesday", "wed", "thursday", "thurs", "thur", "friday", "fri", "saturday", "sat", 
				"sunday", "sun", "yesterday", "today"};
		
		type1aspects.put("when", keywordsForWhen);

		String[] keywordsForWhere = {"location", "locality", "point", "position", "site", "spot", "place", 
				"nearby", "close to", "around", "adjacent to", "intersection", "street", "road", "highway",
				"boulevard", "blvd", "interstate", "outside", "inside", "near", "landfall", "epicenter"};
		type1aspects.put("where", keywordsForWhere);

		String[] keywordsForWhy = {"broken", "faulty", "careless", "inattention", "bad judgment", "poor judgment", 
				"fatigue", "faulty equipment", "defective equipment", "equipment failure", 
				"weather related", "weather", "design", "cause", "explanation", "explain", 
				"reason", "precipitate", "distract", "drug", "alcohol", 
				"due to", "thermonuclear activity", "seismic activity", "seismic", 
				"plate movement", "tectonic movement", "plate", "tectonic", "movement", 
				"human exploration", "human involvement", "human inteference"};
		type1aspects.put("why", keywordsForWhy);
		

		String[] keywordsForWhoaffected = {"casualty",  "dead", "wound", "injur", "loss of life",
				"demise", "mortality", "survivor", "casualt", "death toll", 
				"wound", "injur", "loss of life", "decease",
				"hurt", "harm", "death toll", "death", "missing", "threat"};
		type1aspects.put("whoaffected", keywordsForWhoaffected);

		String[] keywordsForDamages = { "property", "possession", "belongings", "goods", "assets", "fiscal", 
				"money", "economic", "financial", "loss", "damage", "ruin", "burn", "property loss",
				"financial loss", "property", "possession", "belongings", "goods", 
				"assets", "fiscal", "money", "economic", "property loss", 
				"financial loss","financial", "loss", "damage to", "disruption of"};
		type1aspects.put("damages", keywordsForDamages);

		String[] keywordsForCountermeasures = {"safety", "prevent", "rescue", "hazardous condition",
				"unsafe condition", "hazardous", "unsafe", "maintenance", "repair", 
				"correct", "recovery plan", "prevent", "prevention effort", "rescue effort", 
				"relief effort", "relief", "emergency plan", 
				"risk reduction", "preparedness"};
		type1aspects.put("countermeasures", keywordsForCountermeasures);
		return type1aspects;
	}

	private static Map<String, String[]> generateAspectsFor2() {
		type2aspects = new LinkedHashMap<String, String[]>();
		String[] keywordsForWhat = { "felony", "lawlessness", "malfeasance", "misdemeanor", "transgression",
				"unlawful", "violation", 
				"wrongdoing", "raid", "ambush", "assault", "insurrection", "violence", "calculated", "threat", 
				 "riot", "violence", "raid", "ambush", "assault", "revolt", "violence", "massacre", "slaughter",
				 "assault", "battery", "burglary", "theft", "home invasion", "murder", "rape", 
				 "child molestation", "molestation", 
				 "sexual abuse", "abuse", "robbery", "manslaughter", "kidnapping", "abduction", 
				 "poisoning", "shooting", 
				 "endangerment", "hate crime", "political", "religious", "guerilla", "nuclear", 
				 "chemical weapon", "biological weapon", 
				 "chemic", "biologic", "weapon", "weapon of mass destruction", "bomb", "mass destruction",
				 "bomb", "gunfire", "landmine", "explosion", "nuclear", "uprising"};
		type2aspects.put("what", keywordsForWhat);

		String[] keywordsForWhen = {"january", "jan", "february", "feb", "march", "mar", "april", "apr", "may", "june", 
				"jun", "july", "jul", "august", "aug", "september", "sept", "sep", "october", "oct", "november", "nov", 
				"december", "dec", "monday", "mon", "tuesday", "tues", "tue", "wednesday", "wed", "thursday", "thurs", 
				"thur", "friday", "fri", "saturday", "sat", "sunday", "sun", "yesterday", "today"};
		type2aspects.put("when", keywordsForWhen);

		String[] keywordsForWhere = { "location", "locality", "point", "position", "site", "spot", "place", "nearby", 
				"close to", "around", "adjacent to", "street", "road", "highway", "boulevard", "blvd", "interstate", 
				"outside", "inside", "near"};
		type2aspects.put("where", keywordsForWhere);

		String[] keywordsForWhy = { "motive", "reason", "commit", "intent", "mens rea", "cause", "means", "opportunity",
				"politic", "religio", "ideologic", "revenge", "retaliation", "disrupt", "disturb", "instability", "unstable",
				};
		type2aspects.put("why", keywordsForWhy);

		String[] keywordsForWhoaffected = { "casualty", "death", "injur", "dead", "loss", "wound", "hurt", "hostage",
				"casualt",  "damage",  "missing",  "harm", "captive"};
		type2aspects.put("whoaffected", keywordsForWhoaffected);

		String[] keywordsForDamages = {"property", "possession", "belongings", "goods", 
				"assets", "fiscal", "money", "economic", "financial", "loss", "damage", "ruin", "burn", 
				"smash", "destroy", "arson", "property loss", "financial loss", "property", "possession", "belongings", "goods", 
				"assets", "fiscal", "money", "economic", "financial", "loss", "damage to", "disruption of", "damage", 
				"ruin", "burn", "damage to", "disruption of", "property loss", 
				"financial loss", "property", "possession", "belongings", 
				"goods", "assets", "fiscal", "money", "economic", "financial", 
				"loss", "damage to", "disruption of", 
				"damage", "ruin", "burn", "damage to", "disruption of",  "property loss", "financial loss"};
		type2aspects.put("damages", keywordsForDamages);

		String[] keywordsForCountermeasures = {"secur",  "prevent", "police", 
				"light", "surveillance camera", "surveillan", "camera", "patrol", "aware", 
				"aware", "suspicious behavior", "suspicious", "unusual", "out of the ordinary", "watch list",
				"secur", "prevent", "patrol", "aware", "surveillance", "police response","risk analysis", "police", "suspicious", "unusual"};
		type2aspects.put("countermeasures", keywordsForCountermeasures);

		String[] keywordsForPerpetrators = {"perpetrate", "commit", "culprit", "criminal", "incendiar", "wrongdoer", 
				"guilty party", "agent", "offender", "perp", "thug", "vandal", "accused", "suspect",
				"terrorist", "subversive", "bomber", "guerilla", "incendiar", "radical", "rebel", "revolutionary", 
				"jihadist", "anarchist", "insurgent", "attacker", "aggressor", "assailant", "assaulter", "mugger"};
		type2aspects.put("perpetrators", keywordsForPerpetrators);
		return type2aspects;
	}

	private static Map<String, String[]> generateAspectsFor3() {
		type3aspects = new LinkedHashMap<String, String[]>();
		String[] keywordsForWhat = {"fitness", "physical", "condition", "wellness", "wellbeing", "welfare", 
				"shape", "disease", "disorder", "guard", "injury", "prevent", "accident", "guard", 
				"protect", "precaution", "safeguard", "prevention", "public", "child", "animal", 
				"community", "senior", "geront", "mental", "accident", "attack", "disaster"};
		type3aspects.put("what", keywordsForWhat);

		String[] keywordsForWhy = {"malnutrition", "disease", "poverty", "infectious disease", "preventable disease", "cardiovascular disease", 
				 "obesity", "medical care", "medical access", "lack of sanitation", 
				"unsanitary", "due to", "attack", "accident", "disaster", "careless", "fatigue", "shortcuts", "due to"};
		type3aspects.put("why", keywordsForWhy);

		String[] keywordsForWhoaffected = {"dead", "ill", "family", "relative", "community", "population", "elder", 
				"young", "children", "infants", "aged", "compromised", "injure", "survivor", "community"};
		type3aspects.put("whoaffected", keywordsForWhoaffected);

		String[] keywordsForHow = {"illness", "injur", "sick", "incapacit", "invalid", "death", "malnourish", "obesity", 
				"loss of life", "coma", "unconscious", "mortalit",  "injur", "death", "dismemberment", "loss of life", "loss of limb"};
		type3aspects.put("how", keywordsForHow);

		String[] keywordsForCountermeasures = {"educate", "prevent", "maintenance", "universal health care", "primary health care", "health care", "immuniz", "medic", 
				 "repair", "educat"};
		type3aspects.put("countermeasures", keywordsForCountermeasures);
		return type3aspects;
	}

	private static Map<String, String[]> generateAspectsFor4() {
		type4aspects = new LinkedHashMap<String, String[]>();
		String[] keywordsForWhat = {"destruction", "dying", "vulnerable", "extinction", "depletion", "lessen", "disappear", 
				"decrease", "threat", "conservation", "environment", "forest", "mineral", "oil", "gas", "coal", "marsh", "wetland", 
				"reef", "river", "land", "water supply", "waterway", "water", "native plant", "native", "endangered speci", "endangered animal", 
				"endangered plant", "endanger", "plant", "animal kingdom", "animal", "fish", "wildlife", 
				"wildflowers", "habitat", "culture", "biodiversity"};
		type4aspects.put("what", keywordsForWhat);

		String[] keywordsForImportance = {"sustain", "cure", "economic", "medical discover", "economic development",  "health", "disaster resilience"};
		type4aspects.put("importance", keywordsForImportance);

		String[] keywordsForThreats = {"disaster", "weather", "pollution", "nonnative speci", "nonnative", "invasive speci", "invasive", "market demand", "urban expansion", 
				"expansion", "natural disaster",  "global warming", "climate change", "war", "conflict", 
				"deforestation", "acid rain", "greenhouse gas", "greenhouse effect", "habitat destruct"};
		type4aspects.put("threats", keywordsForThreats);

		String[] keywordsForCountermeasures = {"cloning", "conservation", "preservation", "wildlife refuge", "recycling", "environmental protection", 
				"habitat protection", "environment", "protect", "behavior", "renewable energy", "activism", "educat", "breeding program", "zoo", "habitat"};
		type4aspects.put("countermeasures", keywordsForCountermeasures);
		return type4aspects;
	}

	private static Map<String, String[]> generateAspectsFor5() {
		type5aspects = new LinkedHashMap<String, String[]>();
		String[] keywordsForWhat = {"inquiry", "enquiry", "inquest", "probe", "check",
				"hearing", "legal proceeding", "tribunal", "suit", "court", "case",
				"criminal", "political", "civil", "negligence", "corporat", "background", 
				"personal", "security", "forensic", "criminal", "civil", "war crimes", "war"};
		type5aspects.put("what", keywordsForWhat);
		
		
		String[] keywordsForWho = {"suspect", "accuse", "culprit", "offend", "violat", 
				"defendant", "plaintiff", "convict", "defense", "appellant"};
		type5aspects.put("who", keywordsForWho);

		String[] keywordsForWho_Inv = {"auditor", "commissioner", "constable", "county clerk", 
				"deputy", "inspectory", "marshal", "police officer", "police", "officer", "private detective", 
				"private investigator", "detective", "investigat", "sheriff", "undercover", "magistrate", 
				"coroner", "controller", "comptroller", "judge", "constable", "magistrate", "tribunal", 
				"arbitrator", "adjudicator", "judiciary", "court"};
		type5aspects.put("who_inv", keywordsForWho_Inv);

		String[] keywordsForWhy = {"accident",  "death", "crime", "corruption", "theft", "fire", "abuse", "due to",
				"determine guilt", "determine innocen", "guilt", "innocen", "wrongdoing", "charge", "justice", "verdict"};
		type5aspects.put("why", keywordsForWhy);

		String[] keywordsForCharges = {"accus", "blame", "wrongdoing", "crime", "conspiracy", "violence", "drug", "alcohol", "offense", 
				"victim", "steal", "theft", "weapon", "crime", "felony", "misdemeanor", "arson", "assault", "burglary", "murder", 
				"rape", "abuse", "robbery"};
		type5aspects.put("charges", keywordsForCharges);

		String[] keywordsForPlead = {"innocent", "denial", "admission", "wrongdoing", "circumstance", "explain", 
				"explanation", "charges", "guilty", "not guilty", "no contest", "insanity", "nolo contendere", "failure to appear", 
				"refus", "plea bargain", "drop charges", "reduce charges", "reduction of charges", "drop", "reduc", 
				"charges"};
		type5aspects.put("plead", keywordsForPlead);

		String[] keywordsForSentence = {"admonishment", "fine", "charge", "reprimand", "counsel", "recommend", "forfeiture",
				"community service", "life sentence", "death sentence", "life", "death", "sentence", "fine", "forfeiture", 
				"imprison", "incarcerat", "jail time", "month", "year", "probation", "treatment", "monitor", 
				"counsel", "reprimand"};
		type5aspects.put("sentence", keywordsForSentence);
		return type5aspects;
	}

	private Category(int idx, Map<String, String[]> aspects) {
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
