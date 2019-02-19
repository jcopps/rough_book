# -*- coding: utf-8 -*- 

import xlrd
import xlwt
from xlwt import Workbook
import json 
from json import dumps, loads, JSONEncoder, JSONDecoder
import pickle

import re
from pathlib import Path

xls_path = '/Users/jcopps/Documents/Stuff/PT Translated/'
TRANS_F = ['Batch_01162018_pt.xlsx', 'Batch_02052018_pt.xlsx', 'Batch_02132018_pt.xlsx']
F2 = 'Batch_02052018_pt.xlsx'
F3 = 'Batch_02132018_pt.xlsx'
eng_train_json_path = r'/Users/jcopps/Documents/Stuff/PT Translated/training/latest/training_data_eng.json'
inf_path = 'influential_features_articles.json'
sheet_name = 'Templates'
inf_path = '/Users/jcopps/code/vtv/src/Conversation/kramer/phrase_server/multiLangNLP/influential_features_articles.json'


#workbook = xlrd.open_workbook(xls_path)

ENG_TMP = 'eng_template'
PT_TMP = 'pt_template'
tagged_query_s = "tagged_query"
tag_type1_s = 'tag_type1'
tag_type2_s = 'tag_type2'
ENTITY_ = 'ENTITY_'
ENTITY = "ENTITY"
typ_s = 'TYP'
filter_s = 'filter'
g_s = 'G'
tag_type_s = 'TAG_TYPE'
sidx_s = 'Sidx'
star_type_s = 'STAR_TYPE'
phrase_s = 'Phrase'
eng_s = 'eng'
por_s = 'por'
meta_s = 'meta_s'

influential_feature_s = 'INFLUENTIAL_FEATURES'


class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (list, dict, str, unicode, int, float, bool, type(None))):
            return JSONEncoder.default(self, obj)
        return {'_python_object': pickle.dumps(obj)}

def main():



    file_meta = { eng_s : 7, por_s : 0}
    file_info = { 'path':'/Users/jcopps/Documents/Stuff/PT Translated/Batch_01162018_pt.xlsx', 'sheet_name': 'Important Features', meta_s : file_meta}
    file_info_list = [file_info]

    file_meta = { eng_s : 2, por_s : 4}
    file_info = { 'path':'/Users/jcopps/Documents/Stuff/PT Translated/Batch_02052018_pt.xlsx', 'sheet_name': 'Filters', meta_s : file_meta}	

    file_info_list.append(file_info)

    eng_to_flang_table = get_translated_file(file_info_list)

    eng_to_flang = table_to_dict(eng_to_flang_table)

    #print "ENG_FLANG: ", eng_to_flang

    inf_json = training_json_helper('').load_inf_dict(inf_path)

    por_dict = inf_json.get(eng_s, '')

    #print por_dict

    for key, value in por_dict.iteritems():
	if key in ['DISSIMILAR_PAIRS', 'SIMILAR_PAIRS']:
	    por_dict[key] = []
	    continue
        print key,"::"	
	if type(value) == type(set()):
	    por_set = []	
	    for feat in value:
		
		
		por_v = eng_to_flang.get(feat, "#"+feat+"#")
	        if por_v not in por_set:
                    por_set.append(por_v)
	    por_dict[key] = por_set
	
	else:
	    dict_obj = {}
	    for key_2, value_2 in value.iteritems():
		
		por_set = []
		for feat in value_2:
		   translated_word = eng_to_flang.get(feat, "#"+feat+"#")
		   if translated_word not in por_set: 
		       por_set.append(eng_to_flang.get(feat, "#"+feat+"#"))		
		
		dict_obj[key_2] = por_set

	    por_dict[key] = dict_obj
		

    eng_to_flang[por_s] =  por_dict
    #print eng_to_flang[por_s]
    j = dump_dict(eng_to_flang[por_s], "inf.json")
    print eng_to_flang
    #print j
    #print eng_to_flang[por_s]['INFLUENTIAL_FEATURES']

def table_to_dict(table):

    dict_obj = {}
    for record in table:
	key_lower = record[eng_s].lower()
	if key_lower not in dict_obj:
		dict_obj[key_lower] = record[por_s]
	

    return dict_obj

class xls_helper:
    def __init__(self, file_name, sheet_name):
	self.table_dict = None
	workbook = xlrd.open_workbook(file_name)
	self.worksheet = workbook.sheet_by_name(sheet_name)

    def process_xls(self, include_header = False):
	
        self.n_rows = self.worksheet.nrows
	self.n_cols = self.worksheet.ncols
	#print self.worksheet.cell(0, 0).value, self.worksheet.cell(0, 1).value
	table_d = {}
	for row in range(1 if include_header == False else 0, self.n_rows): #start from 1, to leave out row 0
	    table_d[row] = []
	    for col in range(0, self.n_cols):
		table_d[row].append(self.worksheet.cell(row, col).value)

        self.table_dict = table_d
	return self.table_dict
    
    def select_cols(self, cols_dict):
	select_cols = []

	if self.table_dict is None:
	    self.process_xls()

	for row_index, row_data in self.table_dict.iteritems():
	    select_row = {}
	    for col_name, col_index in cols_dict.iteritems():

		if col_index < self.n_cols:
		    select_row[col_name] = self.table_dict[row_index][col_index]
	    select_cols.append(select_row)
	return select_cols    

def dump_dict(dict_obj, file_loc):
    json_obj= (json.dumps(dict_obj, ensure_ascii=False).encode('utf8'))
    #print json_obj
    if file_loc is not None:
        file = open(file_loc, "w")
        file.write(json_obj)
        file.close()
    else:
        print dict_obj
'''
def load_dict(file_loc):
    if(os.path.isfile(file_loc)):
        file = open(file_loc, "r")
        content = file.read()
        dict_obj = json.loads(content)
        return dict_obj
    else:
        return False
'''

class template_q:
    def __init__(self):
        self.freq = 0
        #self.type_freq = {}
	self.info_l = []

    def add(self, info_l):
        self.freq += 1
	if len(self.info_l) == 0: # Remove this condition for convo
	    self.info_l.append(info_l)

class info:
    def __init__(self):
	self.connections = []
	self.filters = []
	self.intent = []
	self.deep_descriptors = []
	self.intent = []
	self.sub_intent = []
	self.action = []
	self.inf_feat = []


    
class training_json_helper:
    def __init__(self, train_path):
        self.train_path = train_path
	self.feat_list = None
       	self.infl_dict = None

    def get_content_type(self, query):

        return query[query.find(ENTITY_)+len(ENTITY_):].split(None, 1)[0]

    def trim_entity_type(self, entity_type):
	return entity_type[entity_type.find(ENTITY_)+len(ENTITY_):]

    def process_files(self, eng_flang_dict):

        file = open(self.train_path, "r")
        content = file.read()
        train_json = json.loads(content)
        
	test_cases = train_json["test_cases"]
        self.template_dict = {}
	index = 1
        for case_no, testcase in test_cases.iteritems():
	    
	    inter_info = info()
	    case_obj = testcase[0]
            tagged_query = case_obj[tagged_query_s]

	    connections = case_obj['connections']
	    if len(connections) > 0:
		for conn in connections:
		    if conn.has_key(tag_type1_s) and conn.has_key(tag_type2_s):
			inter_info.connections.append((self.trim_entity_type(conn[tag_type1_s]), self.trim_entity_type(conn[tag_type2_s])))


	    if case_obj.has_key('entities'):
		
		filter_list = []
		deep_descr_l = []
			
		for entity_elt in case_obj['entities']:

		    if entity_elt.has_key('Entity'):
			entity_list = entity_elt['Entity']
			if len(entity_list) > 0:
			    
			    for entity in entity_list:
				if entity.get(typ_s) == filter_s:
				    #Create a filter obj
				    filter_obj = { g_s : entity.get(g_s), tag_type_s : entity.get(tag_type_s), phrase_s : entity.get(phrase_s)}
				    filter_list.append(filter_obj)
				if entity.has_key(typ_s):	
			            deep_descr_l.append({ sidx_s : entity.get(sidx_s), star_type_s : entity.get(star_type_s), typ_s : entity.get(typ_s), tag_type_s : entity.get(tag_type_s)})
		inter_info.filters = filter_list
    		inter_info.deep_descriptors = deep_descr_l
			    
	    inf_feat_l = self.get_template_inf_feat(tagged_query)
	    inter_info.inf_feat = inf_feat_l
	    #print tagged_query
	    #print inf_feat_l
	    #print "\n"

	    intent = case_obj['intent']
	    if len(intent) > 0:
		inter_info.intent = intent

	    subintent = case_obj['subintent']
	    if len(subintent) > 0:
		inter_info.sub_intent.append(subintent)

	
	    action = case_obj['action']
	    if len(action) > 0:
		inter_info.action = action

	      
	    self.template_dict.setdefault(tagged_query, template_q())
	    self.template_dict[tagged_query].add(inter_info)
	    
	    
	    index += 1
	
	#self.show_template_dict()
	self.generate_xls('intermediate', 'pt', eng_flang_dict)
	#self.generate_missing(eng_flang_dict)
	return self.template_dict

    def get_bigram(self):
	return ['tele', 'trailer', 'stand', 'score', 'winner', 'tv series', 'tv show', 'leaderboard', 'medal', 'alarm', 'team', 'athlete', 'champion', 'switch to', 'stock price', 'musician', 'lose', 'half way mark', 'lectures', 'captain', 'qualify', 'character', 'league', 'results', 'table', 'playing', 'crew', 'match', 'highlight', 'live', 'videos', 'players', 'guest', 'tally', 'actor', 'roles', 'lead', 'cloud', 'the show']

    def has_inf_feature(self, template_str, intent = None):
        bigram_l = get_bigram()
	inf_feat_l = self.get_inf_feature_l(intent)
        tokens_l = template_str.split(" ")

        #print inf_feat_l, " vs ", tokens_l
        if len(list(set(inf_feat_l) & set(tokens_l))) > 0:
            return True
        for bi in bigram_l:
            if template_str.find(bi) > -1:
                return True
        return False
    

    def get_template_inf_feat(self,tagged_query):
	temp_inf_l = []

	inf_feat_l = self.get_inf_feature_l()	

	tokens_l = tagged_query.split(" ")
	temp_inf_l = list(set(tokens_l) & set(inf_feat_l))
	gen_template = re.sub("(ENTITY_)[a-z_]+", "", tagged_query)
	bigrams_l = self.get_bigram()
	#print tagged_query, "::", gen_template
	for bigram in bigrams_l:
	    if bigram in gen_template:
		#print bigram, gen_template
		temp_inf_l.append(bigram)

        return temp_inf_l

    def load_inf_dict(self,path):
        fd = open(path, "r")
        content = fd.read()
        #print content.replace("\\", '')
        inf_dict = eval(content)
        return inf_dict

    def get_inf_feature_l(self,intent = None, lang = eng_s):
        
	if self.feat_list is not None:
	    return self.feat_list

	if self.infl_dict is None:
            self.infl_dict = self.load_inf_dict(inf_path)
        inf_feat_dict = self.infl_dict[lang]
        inf_feat_dict = inf_feat_dict[influential_feature_s]
        
        feat_list = []
        if intent is not None:
            for intent_s in intent:
                if inf_feat_dict.has_key(intent_s):
                    feat_list = list(set(feat_list) | set(inf_feat_dict.get(intent_s)))
        else:
            for key, _list in inf_feat_dict.iteritems():
                feat_list = list(set(feat_list) | set(_list))
        
	self.feat_list = feat_list
        return feat_list

    def create_xls_header(self, ws, headers_list):
        style0 = xlwt.easyxf('font: name Times New Roman size 20, color-index black, bold on')
        for i, header in enumerate(headers_list):
            ws.write(0, i, header, style0)
        return ws

    def write_results_to_xls(self, ws, lang, eng_flang_dict):
	i = 1 # Header - 0
	print eng_flang_dict.keys()[0]
	for tagged_query, template_q in self.template_dict.iteritems():
	    for info in template_q.info_l:
		if tagged_query not in eng_flang_dict.keys():
		    continue
		#print tagged_query, "::", eng_flang_dict[tagged_query]
		ws.write(i, 0, tagged_query)
		ws.write(i, 1, eng_flang_dict.get(tagged_query)[PT_TMP])
		ws.write(i, 2, template_q.freq)
		ws.write(i, 3, json.dumps(info.connections))
		ws.write(i, 4, json.dumps(info.filters))
		ws.write(i, 5, json.dumps(info.intent))
		ws.write(i, 6, json.dumps(info.sub_intent))
		ws.write(i, 7, json.dumps(info.deep_descriptors))
		ws.write(i, 8, json.dumps(info.action))
		ws.write(i, 9, json.dumps(info.inf_feat))
		i += 1
	

    def write_orphan_flang(self, ws, eng_flang_dict):
	i = 1
	for eng_str, flang_str in eng_flang_dict.iteritems():
	    if eng_str not in self.template_dict.keys():
		if i == 1:
		    self.create_xls_header(ws, flang_str.keys())
		j = 0
		for eng_translation, elt in flang_str.iteritems():
		    ws.write(i, j, elt)
		    j += 1
		#print type(eng_str),"::", type(flang_str)
		i += 1

    def generate_missing(self, eng_flang_dict):
	workbook = Workbook(encoding="utf-8")
	ws = workbook.add_sheet('pt_orphan_templates')
        
        self.write_orphan_flang(ws, eng_flang_dict)
        workbook.save('template_info_pt_03.xls')

    def generate_xls(self, sheet_name, lang, eng_flang_dict):
	workbook = Workbook(encoding="utf-8")
	ws = workbook.add_sheet(sheet_name)

	headings = ['Template en', 'Template '+lang, 'Template frequency', 'Connections', 'Filters', 'Intent', 'Subintent', 'Deep descriptors', 'Action', 'Influential feature']

	self.create_xls_header(ws, headings)
	self.write_results_to_xls(ws, lang, eng_flang_dict)
	workbook.save('template_info_pt.xls')


    def show_template_dict(self):
	for tagged_query, template_q in self.template_dict.iteritems():
	    print 'Template en: ', tagged_query
	    print 'Template frequency: ', template_q.freq
	    for info in template_q.info_l:
		print 'Deep descriptor: ', info.deep_descriptors
		print 'Filters: ', info.filters
		print 'Intent: ', info.intent
		print 'Subintent: ', info.sub_intent
		print 'Connections: ', info.connections
	        print '\n'
	print "Templates length: ", len(self.template_dict)

def get_translated_file(file_info_list):
    eng_to_flang_dict = {}

    xls_table = []
    for file_info in file_info_list:
	xls_table = xls_table + translated(file_info['path'], file_info['sheet_name'], file_info[meta_s]).xls_table
	
    if xls_table:
	return xls_table
    else:
	return None


class translated:
    def __init__(self, translated_file_loc, sheet_name, col_indexes={ENG_TMP:1, PT_TMP: 2}):
	
        self.trans_xls_path = translated_file_loc
	self.xls_table = xls_helper(self.trans_xls_path, sheet_name).select_cols(col_indexes)

if __name__ == '__main__':

    main()
