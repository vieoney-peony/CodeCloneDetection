import javalang

class JavaASTNode:
    def __init__(self, node: javalang.tree.Node, sub_index: int = 0):
        self.node = node
        self.label = type(node).__name__
        self.sub_index = sub_index

    def __str__(self):
        return f"{self.label}_{self.sub_index}"
    
    def __repr__(self):
        return str(self)

class JavaASTLiteralNode(JavaASTNode):
    def __init__(self, node: javalang.tree.Node, sub_index: int = 0, value: str = ""):
        super().__init__(node, sub_index)
        self.value = value

class JavaASTBinaryOpNode(JavaASTNode):
    def __init__(self, node: javalang.tree.BinaryOperation, sub_index: int = 0):
        super().__init__(node, sub_index)
        self.operator = node.operator  # Toán tử của phép toán

    def __str__(self):
        return f"{self.label}_({self.operator})_{self.sub_index}"

class JavaASTGraphVisitor:
    def __init__(self):
        # Danh sách lưu các cạnh dưới dạng (parent_node, edge_label, child_node)
        self.edges = []
        # Stack lưu các JavaASTNode của node cha trong quá trình duyệt
        self.parent_stack = []
        # Mapping từ id(node) của javalang sang JavaASTNode
        self.node_mapping = {}
        # Lưu thông tin biến (name -> VariableDeclarator)
        self.variable_map = {}
        # Bộ đếm toàn cục để gán subindex cho mỗi node theo thứ tự duyệt
        self.visitation_counter = 0

    def visit(self, node):
        """Nếu node là list, duyệt từng phần tử; nếu là Node, thực hiện dispatch đến visit_<NodeType>."""
        if isinstance(node, list):
            for n in node:
                self.visit(n)
            self.visit_list(node)
            return
        
        if not isinstance(node, javalang.tree.Node):
            return
        # print('---',type(node))
        sub_index = self.visitation_counter
        self.visitation_counter += 1

        current_obj = JavaASTNode(node, sub_index=sub_index)
        self.node_mapping[id(node)] = current_obj

        if self.parent_stack:
            parent_obj = self.parent_stack[-1]
            self.edges.append((parent_obj, "Child", current_obj))
            self.edges.append((current_obj, "Parent", parent_obj))

        self.parent_stack.append(current_obj)

        method_name = "visit_" + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        visitor(node)

        self.parent_stack.pop()

    def generic_visit(self, node):
        """Duyệt tất cả các children của node."""
        if not isinstance(node, javalang.tree.Node):
            return
        
        children = node.children
        
        for child in children:
            if child is not None:
                self.visit(child)

    def visit_list(self, node):
        """Add Nextsib node"""
        children = [x for x in node if isinstance(x, javalang.tree.Node)]
        for i in range(len(children) - 1):
            current_obj = self.node_mapping.get(id(children[i]))
            next_obj = self.node_mapping.get(id(children[i + 1]))
            self.edges.append((current_obj, "NextSib", next_obj))

    def visit_IfStatement(self, node):
        self.generic_visit(node)

        condition = getattr(node, 'condition', None)
        then_stmt = getattr(node, 'then_statement', None)
        else_stmt = getattr(node, 'else_statement', None)

        cond_obj = self.node_mapping.get(id(condition)) if condition else None
        then_obj = self.node_mapping.get(id(then_stmt)) if then_stmt else None
        else_obj = self.node_mapping.get(id(else_stmt)) if else_stmt else None

        if cond_obj and then_obj:
            self.edges.append((cond_obj, "CondTrue", then_obj))
        if cond_obj and else_obj:
            self.edges.append((cond_obj, "CondFalse", else_obj))

    def visit_WhileStatement(self, node):
        self.generic_visit(node)

        condition = getattr(node, 'condition', None)
        body = getattr(node, 'body', None)

        cond_obj = self.node_mapping.get(id(condition)) if condition else None
        body_obj = self.node_mapping.get(id(body)) if body else None

        self.edges.append((cond_obj, "WhileExec", body_obj))
        self.edges.append((body_obj, "WhileNext", cond_obj))
    
    def visit_BlockStatement(self, node):
        self.generic_visit(node)

        statements = getattr(node, 'statements', [])
        for i in range(len(statements) - 1):
            stmt_obj = self.node_mapping.get(id(statements[i]))
            next_stmt_obj = self.node_mapping.get(id(statements[i + 1]))
            self.edges.append((stmt_obj, "NextStmt", next_stmt_obj))
        
    def visit_ForStatement(self, node):
        # print(node.attrs)
        self.generic_visit(node)
        control = getattr(node, 'control', None)
        body = getattr(node, 'body', None)

        control_obj = self.node_mapping.get(id(control)) if control else None
        body_obj = self.node_mapping.get(id(body)) if body else None

        self.edges.append((control_obj, "ForExec", body_obj))
        self.edges.append((body_obj, "ForNext", control_obj))
          
    def visit_DoStatement(self, node):
        self.generic_visit(node)

        condition = getattr(node, 'condition', None)
        body = getattr(node, 'body', None)

        cond_obj = self.node_mapping.get(id(condition)) if condition else None
        body_obj = self.node_mapping.get(id(body)) if body else None

        self.edges.append((cond_obj, "DoExec", body_obj))
        self.edges.append((body_obj, "DoNext", cond_obj))
    
    def visit_SwitchStatement(self, node):
        self.generic_visit(node)

        expression = getattr(node, 'expression', None)
        cases = getattr(node, 'cases', [])

        expr_obj = self.node_mapping.get(id(expression)) if expression else None
        for case in cases:
            case_obj = self.node_mapping.get(id(case))
            self.edges.append((expr_obj, "SwitchExec", case_obj))

    def visit_VariableDeclarator(self, node):
        """Xử lý khai báo biến"""
        var_name = node.name  # Tên biến
        current_obj = self.node_mapping.get(id(node)) # JavaASTNode
        if var_name in self.variable_map:
            self.variable_map[var_name].append(current_obj)
        else:
            self.variable_map[var_name] = [current_obj] # list of JavaASTNode
        self.generic_visit(node)
        # self.variable_map[var_name].pop()
        
    def visit_MemberReference(self, node):
        self.generic_visit(node)
        
        var_name = node.member
        if node.qualifier:
            # print(var_name, node.qualifier, type(node.qualifier), )
            var_name = node.qualifier + '.' + var_name

        current_obj = self.node_mapping.get(id(node)) # JavaASTNode
        
        if var_name in self.variable_map:
            var_decl_label_list = self.variable_map[var_name] # list of JavaASTNode
            # if var_decl_label_list == []:
            #     print(var_name)
            #     print(node)
            var_decl_label = var_decl_label_list[-1]
            self.edges.append((var_decl_label, 'NextUse', current_obj))
        
            self.variable_map[var_name].append(current_obj)
        else:
            self.variable_map[var_name] = [current_obj]

    def visit_BinaryOperation(self, node):
        """Xử lý phép toán nhị phân (BinaryOperation)"""
        self.generic_visit(node)  # Duyệt qua toán hạng trước khi cập nhật node

        # Lấy node hiện có từ node_mapping
        binary_node = self.node_mapping.get(id(node))
       
        binary_node.__class__ = JavaASTBinaryOpNode  # Downcast đối tượng
        binary_node.operator = node.operator  # Gán toán tử

        operandl = getattr(node, 'operandl', None)
        operandr = getattr(node, 'operandr', None)

        operandl_obj = self.node_mapping.get(id(operandl)) if operandl else None
        operandr_obj = self.node_mapping.get(id(operandr)) if operandr else None

        self.edges.append((operandl_obj, "LeftOperand", operandr_obj))
        self.edges.append((operandr_obj, "RightOperand", operandl_obj))
    
    def visit_Literal(self, node):
        """Xử lý node là literal"""
        self.generic_visit(node)
        value = node.value
        current_obj = self.node_mapping.get(id(node))
        current_obj.__class__ = JavaASTLiteralNode
        current_obj.value = value
        parent_obj = self.parent_stack[-2]
        self.edges.append((parent_obj, "Value", current_obj))
    
    def visit_FormalParameter(self, node):
        var_name = node.name
        current_obj = self.node_mapping.get(id(node))
        self.variable_map[var_name] = [current_obj]
        # print('FormalParameter:', var_name)
        self.generic_visit(node)
    
    def visit_InferredFormalParameter(self, node):
        # print(node)
        var_name = node.name
        current_obj = self.node_mapping.get(id(node))
        self.variable_map[var_name] = [current_obj]
        # print('FormalParameter:', var_name)
        self.generic_visit(node)


# Ví dụ sử dụng:
if __name__ == "__main__":
    # code = """
    # public class Example {
    #     public static void main(String[] args) {
    #         Data obj = new Data();
    #         int x = obj.sub.value;  // Truy cập a.b.c
            
    #         // Gọi hàm có varargs (informal parameters)
    #         printValues(1, 2, 3, 4, 5);
    #     }

    #     class Data {
    #         SubData sub = new SubData();
    #     }

    #     class SubData {
    #         int value = 42;
    #     }

    #     // Phương thức có varargs (informal parameters)
    #     public static void printValues(int... numbers) {
    #         for (int num : numbers) {
    #             System.out.println(num);
    #         }
    #     }
    # }
    # """
    code = """
        public class DummyClass {
        public static void main(String[] args) {
                LogFrame.getInstance();
                for (int i = 0; i < args.length; i++) {
                    String arg = args[i];
                    if (arg.trim().startsWith(DEBUG_PARAMETER_NAME + "=")) {
                        properties.put(DEBUG_PARAMETER_NAME, arg.trim().substring(DEBUG_PARAMETER_NAME.length() + 1).trim());
                        if (properties.getProperty(DEBUG_PARAMETER_NAME).toLowerCase().equals(DEBUG_TRUE)) {
                            DEBUG = true;
                        }
                    } else if (arg.trim().startsWith(MODE_PARAMETER_NAME + "=")) {
                        properties.put(MODE_PARAMETER_NAME, arg.trim().substring(MODE_PARAMETER_NAME.length() + 1).trim());
                    } else if (arg.trim().startsWith(AUTOCONNECT_PARAMETER_NAME + "=")) {
                        properties.put(AUTOCONNECT_PARAMETER_NAME, arg.trim().substring(AUTOCONNECT_PARAMETER_NAME.length() + 1).trim());
                    } else if (arg.trim().startsWith(SITE_CONFIG_URL_PARAMETER_NAME + "=")) {
                        properties.put(SITE_CONFIG_URL_PARAMETER_NAME, arg.trim().substring(SITE_CONFIG_URL_PARAMETER_NAME.length() + 1).trim());
                    } else if (arg.trim().startsWith(LOAD_PLUGINS_PARAMETER_NAME + "=")) {
                        properties.put(LOAD_PLUGINS_PARAMETER_NAME, arg.trim().substring(LOAD_PLUGINS_PARAMETER_NAME.length() + 1).trim());
                    } else if (arg.trim().startsWith(ONTOLOGY_URL_PARAMETER_NAME + "=")) {
                        properties.put(ONTOLOGY_URL_PARAMETER_NAME, arg.trim().substring(ONTOLOGY_URL_PARAMETER_NAME.length() + 1).trim());
                    } else if (arg.trim().startsWith(REPOSITORY_PARAMETER_NAME + "=")) {
                        properties.put(REPOSITORY_PARAMETER_NAME, arg.trim().substring(REPOSITORY_PARAMETER_NAME.length() + 1).trim());
                    } else if (arg.trim().startsWith(ONTOLOGY_TYPE_PARAMETER_NAME + "=")) {
                        properties.put(ONTOLOGY_TYPE_PARAMETER_NAME, arg.trim().substring(ONTOLOGY_TYPE_PARAMETER_NAME.length() + 1).trim());
                        if (!(properties.getProperty(ONTOLOGY_TYPE_PARAMETER_NAME).equals(ONTOLOGY_TYPE_RDFXML) || properties.getProperty(ONTOLOGY_TYPE_PARAMETER_NAME).equals(ONTOLOGY_TYPE_TURTLE) || properties.getProperty(ONTOLOGY_TYPE_PARAMETER_NAME).equals(ONTOLOGY_TYPE_NTRIPPLES))) System.out.println("WARNING! Unknown ontology type: '" + properties.getProperty(ONTOLOGY_TYPE_PARAMETER_NAME) + "' (Known types are: '" + ONTOLOGY_TYPE_RDFXML + "', '" + ONTOLOGY_TYPE_TURTLE + "', '" + ONTOLOGY_TYPE_NTRIPPLES + "')");
                    } else if (arg.trim().startsWith(OWLIMSERVICE_URL_PARAMETER_NAME + "=")) {
                        properties.put(OWLIMSERVICE_URL_PARAMETER_NAME, arg.trim().substring(OWLIMSERVICE_URL_PARAMETER_NAME.length() + 1).trim());
                    } else if (arg.trim().startsWith(DOCSERVICE_URL_PARAMETER_NAME + "=")) {
                        properties.put(DOCSERVICE_URL_PARAMETER_NAME, arg.trim().substring(DOCSERVICE_URL_PARAMETER_NAME.length() + 1).trim());
                    } else if (arg.trim().startsWith(DOC_ID_PARAMETER_NAME + "=")) {
                        properties.put(DOC_ID_PARAMETER_NAME, arg.trim().substring(DOC_ID_PARAMETER_NAME.length() + 1).trim());
                    } else if (arg.trim().startsWith(ANNSET_NAME_PARAMETER_NAME + "=")) {
                        properties.put(ANNSET_NAME_PARAMETER_NAME, arg.trim().substring(ANNSET_NAME_PARAMETER_NAME.length() + 1).trim());
                    } else if (arg.trim().startsWith(EXECUTIVE_SERVICE_URL_PARAMETER_NAME + "=")) {
                        properties.put(EXECUTIVE_SERVICE_URL_PARAMETER_NAME, arg.trim().substring(EXECUTIVE_SERVICE_URL_PARAMETER_NAME.length() + 1).trim());
                    } else if (arg.trim().startsWith(USER_ID_PARAMETER_NAME + "=")) {
                        properties.put(USER_ID_PARAMETER_NAME, arg.trim().substring(USER_ID_PARAMETER_NAME.length() + 1).trim());
                    } else if (arg.trim().startsWith(USER_PASSWORD_PARAMETER_NAME + "=")) {
                        properties.put(USER_PASSWORD_PARAMETER_NAME, arg.trim().substring(USER_PASSWORD_PARAMETER_NAME.length() + 1).trim());
                    } else if (arg.trim().startsWith(EXECUTIVE_PROXY_FACTORY_PARAMETER_NAME + "=")) {
                        properties.put(EXECUTIVE_PROXY_FACTORY_PARAMETER_NAME, arg.trim().substring(EXECUTIVE_PROXY_FACTORY_PARAMETER_NAME.length() + 1).trim());
                    } else if (arg.trim().startsWith(DOCSERVICE_PROXY_FACTORY_PARAMETER_NAME + "=")) {
                        properties.put(DOCSERVICE_PROXY_FACTORY_PARAMETER_NAME, arg.trim().substring(DOCSERVICE_PROXY_FACTORY_PARAMETER_NAME.length() + 1).trim());
                        RichUIUtils.setDocServiceProxyFactoryClassname(properties.getProperty(DOCSERVICE_PROXY_FACTORY_PARAMETER_NAME));
                    } else if (arg.trim().startsWith(LOAD_ANN_SCHEMAS_NAME + "=")) {
                        properties.put(LOAD_ANN_SCHEMAS_NAME, arg.trim().substring(LOAD_ANN_SCHEMAS_NAME.length() + 1).trim());
                    } else if (arg.trim().startsWith(SELECT_AS_PARAMETER_NAME + "=")) {
                        properties.put(SELECT_AS_PARAMETER_NAME, arg.trim().substring(SELECT_AS_PARAMETER_NAME.length() + 1).trim());
                    } else if (arg.trim().startsWith(SELECT_ANN_TYPES_PARAMETER_NAME + "=")) {
                        properties.put(SELECT_ANN_TYPES_PARAMETER_NAME, arg.trim().substring(SELECT_ANN_TYPES_PARAMETER_NAME.length() + 1).trim());
                    } else if (arg.trim().startsWith(ENABLE_ONTOLOGY_EDITOR_PARAMETER_NAME + "=")) {
                        properties.put(ENABLE_ONTOLOGY_EDITOR_PARAMETER_NAME, arg.trim().substring(ENABLE_ONTOLOGY_EDITOR_PARAMETER_NAME.length() + 1).trim());
                    } else if (arg.trim().startsWith(CLASSES_TO_HIDE_PARAMETER_NAME + "=")) {
                        properties.put(CLASSES_TO_HIDE_PARAMETER_NAME, arg.trim().substring(CLASSES_TO_HIDE_PARAMETER_NAME.length() + 1).trim());
                    } else if (arg.trim().startsWith(CLASSES_TO_SHOW_PARAMETER_NAME + "=")) {
                        properties.put(CLASSES_TO_SHOW_PARAMETER_NAME, arg.trim().substring(CLASSES_TO_SHOW_PARAMETER_NAME.length() + 1).trim());
                    } else if (arg.trim().startsWith(ENABLE_APPLICATION_LOG_PARAMETER_NAME + "=")) {
                        properties.put(ENABLE_APPLICATION_LOG_PARAMETER_NAME, arg.trim().substring(ENABLE_APPLICATION_LOG_PARAMETER_NAME.length() + 1).trim());
                    } else {
                        System.out.println("WARNING! Unknown or undefined parameter: '" + arg.trim() + "'");
                    }
                }
                System.out.println(startupParamsToString());
                if (properties.getProperty(MODE_PARAMETER_NAME) == null || (!(properties.getProperty(MODE_PARAMETER_NAME).toLowerCase().equals(POOL_MODE)) && !(properties.getProperty(MODE_PARAMETER_NAME).toLowerCase().equals(DIRECT_MODE)))) {
                    String err = "Mandatory parameter '" + MODE_PARAMETER_NAME + "' must be defined and must have a value either '" + POOL_MODE + "' or '" + DIRECT_MODE + "'.\n\nApplication will exit.";
                    System.out.println(err);
                    JOptionPane.showMessageDialog(new JFrame(), err, "Error!", JOptionPane.ERROR_MESSAGE);
                    System.exit(-1);
                }
                if (properties.getProperty(SITE_CONFIG_URL_PARAMETER_NAME) == null || properties.getProperty(SITE_CONFIG_URL_PARAMETER_NAME).length() == 0) {
                    String err = "Mandatory parameter '" + SITE_CONFIG_URL_PARAMETER_NAME + "' is missing.\n\nApplication will exit.";
                    System.out.println(err);
                    JOptionPane.showMessageDialog(new JFrame(), err, "Error!", JOptionPane.ERROR_MESSAGE);
                    System.exit(-1);
                }
                try {
                    String context = System.getProperty(CONTEXT);
                    if (context == null || "".equals(context)) {
                        context = DEFAULT_CONTEXT;
                    }
                    String s = System.getProperty(GateConstants.GATE_HOME_PROPERTY_NAME);
                    if (s == null || s.length() == 0) {
                        File f = File.createTempFile("foo", "");
                        String gateHome = f.getParent().toString() + context;
                        f.delete();
                        System.setProperty(GateConstants.GATE_HOME_PROPERTY_NAME, gateHome);
                        f = new File(System.getProperty(GateConstants.GATE_HOME_PROPERTY_NAME));
                        if (!f.exists()) {
                            f.mkdirs();
                        }
                    }
                    s = System.getProperty(GateConstants.PLUGINS_HOME_PROPERTY_NAME);
                    if (s == null || s.length() == 0) {
                        System.setProperty(GateConstants.PLUGINS_HOME_PROPERTY_NAME, System.getProperty(GateConstants.GATE_HOME_PROPERTY_NAME) + "/plugins");
                        File f = new File(System.getProperty(GateConstants.PLUGINS_HOME_PROPERTY_NAME));
                        if (!f.exists()) {
                            f.mkdirs();
                        }
                    }
                    s = System.getProperty(GateConstants.GATE_SITE_CONFIG_PROPERTY_NAME);
                    if (s == null || s.length() == 0) {
                        System.setProperty(GateConstants.GATE_SITE_CONFIG_PROPERTY_NAME, System.getProperty(GateConstants.GATE_HOME_PROPERTY_NAME) + "/gate.xml");
                    }
                    if (properties.getProperty(SITE_CONFIG_URL_PARAMETER_NAME) != null && properties.getProperty(SITE_CONFIG_URL_PARAMETER_NAME).length() > 0) {
                        File f = new File(System.getProperty(GateConstants.GATE_SITE_CONFIG_PROPERTY_NAME));
                        if (f.exists()) {
                            f.delete();
                        }
                        f.getParentFile().mkdirs();
                        f.createNewFile();
                        URL url = new URL(properties.getProperty(SITE_CONFIG_URL_PARAMETER_NAME));
                        InputStream is = url.openStream();
                        FileOutputStream fos = new FileOutputStream(f);
                        int i = is.read();
                        while (i != -1) {
                            fos.write(i);
                            i = is.read();
                        }
                        fos.close();
                        is.close();
                    }
                    try {
                        Gate.init();
                        gate.Main.applyUserPreferences();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    s = BASE_PLUGIN_NAME + "," + properties.getProperty(LOAD_PLUGINS_PARAMETER_NAME);
                    System.out.println("Loading plugins: " + s);
                    loadPlugins(s, true);
                    loadAnnotationSchemas(properties.getProperty(LOAD_ANN_SCHEMAS_NAME), true);
                } catch (Throwable e) {
                    e.printStackTrace();
                }
                MainFrame.getInstance().setVisible(true);
                MainFrame.getInstance().pack();
                if (properties.getProperty(MODE_PARAMETER_NAME).toLowerCase().equals(DIRECT_MODE)) {
                    if (properties.getProperty(AUTOCONNECT_PARAMETER_NAME, "").toLowerCase().equals(AUTOCONNECT_TRUE)) {
                        if (properties.getProperty(DOC_ID_PARAMETER_NAME) == null || properties.getProperty(DOC_ID_PARAMETER_NAME).length() == 0) {
                            String err = "Can't autoconnect. A parameter '" + DOC_ID_PARAMETER_NAME + "' is missing.";
                            System.out.println(err);
                            JOptionPane.showMessageDialog(MainFrame.getInstance(), err, "Error!", JOptionPane.ERROR_MESSAGE);
                            ActionShowDocserviceConnectDialog.getInstance().actionPerformed(null);
                        } else {
                            ActionConnectToDocservice.getInstance().actionPerformed(null);
                        }
                    } else {
                        ActionShowDocserviceConnectDialog.getInstance().actionPerformed(null);
                    }
                } else {
                    if (properties.getProperty(AUTOCONNECT_PARAMETER_NAME, "").toLowerCase().equals(AUTOCONNECT_TRUE)) {
                        if (properties.getProperty(USER_ID_PARAMETER_NAME) == null || properties.getProperty(USER_ID_PARAMETER_NAME).length() == 0) {
                            String err = "Can't autoconnect. A parameter '" + USER_ID_PARAMETER_NAME + "' is missing.";
                            System.out.println(err);
                            JOptionPane.showMessageDialog(MainFrame.getInstance(), err, "Error!", JOptionPane.ERROR_MESSAGE);
                            ActionShowExecutiveConnectDialog.getInstance().actionPerformed(null);
                        } else {
                            ActionConnectToExecutive.getInstance().actionPerformed(null);
                        }
                    } else {
                        ActionShowExecutiveConnectDialog.getInstance().actionPerformed(null);
                    }
                }
            }
        }
    """
    # Parse code Java thành AST bằng javalang
    ast_tree = javalang.parse.parse(code)
    # print(ast_tree)
    # Tạo visitor và duyệt AST
    visitor = JavaASTGraphVisitor()
    visitor.visit(ast_tree)
    # print(ast_tree)
    # print(visitor.variable_map)
    # In ra danh sách các cạnh dưới dạng tuple: (parent, edge_label, child)
    # for edge in visitor.edges:
    #     print(edge)
    
    # Ví dụ in ra các node với subindex
    # print("\nCác node đã duyệt:")
    for node in visitor.node_mapping.values():
        with open('test.txt','a') as f:
            f.write(str(node.node)+'\n')
            f.write('-'*50+'\n')
        # print(node.node)
        # print('-'*50)