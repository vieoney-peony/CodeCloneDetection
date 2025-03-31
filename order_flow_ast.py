import javalang
import logging

logging.basicConfig(filename="graph.txt", level=logging.INFO, format="%(message)s", filemode="w")

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

    def __str__(self):
        return f"{self.label}_{self.sub_index}_{self.value}"

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
        # Lưu thông tin method declaration (name -> MethodDeclaration)
        self.method_map = {}
        # Lưu thông tin type (name -> ReferenceType)
        self.type_map = {}
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
            # print('---', node, type(node))
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
    
    def visit_ClassDeclaration(self, node):
        current_obj = self.node_mapping.get(id(node))
        current_obj.__class__ = JavaASTLiteralNode
        modifiers = str(getattr(node, 'modifiers', None))
        current_obj.value = modifiers
        self.type_map[str(node.name)] = [current_obj]
        self.generic_visit(node)

        
    def visit_MemberReference(self, node):
        var_name = node.member
        if node.qualifier:
            var_name = node.qualifier + '.' + var_name

        current_obj = self.node_mapping.get(id(node)) # JavaASTNode
        
        if var_name in self.variable_map:
            var_decl_label_list = self.variable_map[var_name] # list of JavaASTNode
            var_decl_label = var_decl_label_list[-1]
            self.edges.append((var_decl_label, 'NextUse', current_obj))
            self.variable_map[var_name].append(current_obj)
        else: # reference to a system variable
            current_obj.__class__ = JavaASTLiteralNode
            current_obj.value = var_name
            # self.variable_map[var_name] = [current_obj]

        self.generic_visit(node)

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
        qualifier = str(getattr(node, 'modifiers', ''))
        value = node.value
        current_obj = self.node_mapping.get(id(node))
        current_obj.__class__ = JavaASTLiteralNode
        if qualifier != '':
            qualifier = qualifier + '.'
        current_obj.value = qualifier + '.' +value
        parent_obj = self.parent_stack[-2]
        self.edges.append((parent_obj, "Value", current_obj))
        self.generic_visit(node)
    
    def visit_FormalParameter(self, node):
        var_name = node.name
        current_obj = self.node_mapping.get(id(node))
        self.variable_map[var_name] = [current_obj]
        current_obj.__class__ = JavaASTLiteralNode
        modifiers = str(getattr(node, 'modifiers', None))
        current_obj.value = modifiers
        # print('FormalParameter:', var_name)
        self.generic_visit(node)
    
    def visit_InferredFormalParameter(self, node):
        # print(node)
        var_name = node.name
        current_obj = self.node_mapping.get(id(node))
        self.variable_map[var_name] = [current_obj]
        # print('FormalParameter:', var_name)
        self.generic_visit(node)
    
    def visit_TypeDeclaration(self, node):
        current_obj = self.node_mapping.get(id(node))   
        self.type_map[str(node.name)] = [current_obj]
        self.generic_visit(node)

    def visit_ReferenceType(self, node):
        current_obj = self.node_mapping.get(id(node))
        if node.name in self.type_map:
            self.type_map[str(node.name)].append(current_obj)
        else:
            current_obj.__class__ = JavaASTLiteralNode
            current_obj.value = str(node.name)

        self.generic_visit(node)
    
    def visit_BasicType(self, node):
        current_obj = self.node_mapping.get(id(node))
        current_obj.__class__ = JavaASTLiteralNode
        current_obj.value = str(node.name)
        # print(node)
        # print('##'*50)
        self.generic_visit(node)
    
    def visit_MethodDeclaration(self, node):
        current_obj = self.node_mapping.get(id(node))
        self.variable_map[node.name] = [current_obj]
        current_obj.__class__ = JavaASTLiteralNode
        modifiers = str(getattr(node, 'modifiers', None))
        current_obj.value = modifiers
        self.generic_visit(node)
    
    def visit_VariableDeclaration(self, node):
        current_obj = self.node_mapping.get(id(node))
        current_obj.__class__ = JavaASTLiteralNode
        modifiers = str(getattr(node, 'modifiers', None))
        current_obj.value = modifiers
        self.generic_visit(node)
    
    def visit_LocalVariableDeclaration(self, node):
        current_obj = self.node_mapping.get(id(node))
        current_obj.__class__ = JavaASTLiteralNode
        modifiers = str(getattr(node, 'modifiers', None))
        current_obj.value = modifiers
        self.generic_visit(node)

    def visit_MethodInvocation(self, node):
        current_obj = self.node_mapping.get(id(node))
        member_name = node.member
        
        invoke_name = node.qualifier
        invoke_obj = None
        
        if invoke_name in self.variable_map:
            invoke_obj = self.variable_map[invoke_name][0]
            self.variable_map[invoke_name].append(current_obj)
        elif invoke_name in self.type_map:
            invoke_obj = self.type_map[invoke_name][0]
            self.type_map[invoke_name].append(current_obj)
            
        # if invoke_name == 'arg':
        #     print('arg', current_obj, invoke_obj, invoke_name, member_name)

        if invoke_obj:
            self.edges.append((invoke_obj, "MethodInvoke", current_obj))

        if invoke_name is not None and \
            invoke_name not in self.variable_map and \
                invoke_name not in self.type_map:
            current_obj.__class__ = JavaASTLiteralNode
            current_obj.value = invoke_name
            if member_name not in self.variable_map and member_name not in self.type_map:
                current_obj.value += '.' + member_name
        else:
            if member_name not in self.variable_map and member_name not in self.type_map:
                current_obj.__class__ = JavaASTLiteralNode
                current_obj.value = '.' + member_name

        self.generic_visit(node)
    
    def visit_CatchClauseParameter(self, node):
        """Xử lý tham số của khối catch"""
        var_name = node.name
        current_obj = self.node_mapping.get(id(node))
        current_obj.__class__ = JavaASTLiteralNode
        modifiers = str(getattr(node, 'modifiers', None))
        current_obj.value = modifiers
        self.variable_map[var_name] = [current_obj]
        for i, type in enumerate(node.types):
            if i == 0:
                current_obj.value = current_obj.value + '.' + str(type)
            else:
                current_obj.value = current_obj.value + ' | ' + str(type)

        self.generic_visit(node)

def print_ast(node, indent=0, key="root"):
    """Duyệt đệ quy AST, in cả key và class với định dạng 'key: ClassName'"""
    prefix = " " * indent + f"{key}: {node.__class__.__name__}" if isinstance(node, javalang.ast.Node) else " " * indent + f"{key}: {node}"

    logging.info(prefix)  # In key + class của node

    if isinstance(node, javalang.ast.Node):
        for name, child in node.__dict__.items():
            if isinstance(child, list):  # Nếu là danh sách node con
                for i, item in enumerate(child):
                    print_ast(item, indent + 4, key=f"{name}[{i}]")  
            elif isinstance(child, javalang.ast.Node):  # Nếu là node đơn lẻ
                print_ast(child, indent + 4, key=name)
            else:
                logging.info(f"{' ' * (indent + 4)}{name}: {child}")  # In giá trị đơn giản

# Ví dụ sử dụng:
if __name__ == "__main__":
    code = """
    class Test {
        public static void main(String[] args) {
            int[] arr = {1, 2, 3};
            int x = arr[0];
            int y = arr[1 + 1]; // Truy cập mảng bằng biểu thức
        }
    }
    """
    # code = """
    # public class MethodReferenceTest {
    
    #     // Static method to be referenced
    #     public static int staticParse(String s) {
    #         return Integer.parseInt(s);
    #     }

    #     // Instance method to be referenced
    #     public String instanceParse(String s) {
    #         return s;
    #     }

    #     public static void main(String[] args) {
    #         // 1️⃣ Static Method Reference
    #         Function<String, Integer> staticRef = Integer::parseInt;
    #         System.out.println("Static method reference result: " + staticRef.apply("123"));

    #         // 2️⃣ Instance Method Reference (on an object)
    #         MethodReferenceTest obj = new MethodReferenceTest();
    #         Function<String, Integer> instanceRef = obj::instanceParse;
    #         System.out.println("Instance method reference result: " + instanceRef.apply("456"));

    #         // 3️⃣ Instance Method Reference (on a class)
    #         Function<String, Integer> classRef = String::length;
    #         System.out.println("Instance method on class reference: " + classRef.apply("Hello"));

    #         // 4️⃣ Constructor Reference
    #         Supplier<StringBuilder> constructorRef = StringBuilder::new;
    #         System.out.println("Constructor reference created: " + constructorRef.get());

    #         // 5️⃣ Method Reference with custom instance method
    #         Function<String, Integer> customRef = obj::instanceParse;
    #         int a = customRef("001");
    #         File attachmentsDir = new File(datadir);
    #         System.out.println("Custom method reference: " + customRef.apply("789"));
    #         System.out.println(Integer.MAX_VALUE);
    #     }
    # }
    # """
    # Parse code Java thành AST bằng javalang
    from datasets import load_from_disk
    jsonl_dataset = load_from_disk('Processed_BCB_code')
    # print(jsonl_dataset['func'][0])
    # ast_tree = javalang.parse.parse(jsonl_dataset['func'][300])
    ast_tree = javalang.parse.parse(code)
    print_ast(ast_tree)
    # Tạo visitor và duyệt AST
    visitor = JavaASTGraphVisitor()
    visitor.visit(ast_tree)
    # print(ast_tree)
    logging.info(visitor.variable_map)
    logging.info(visitor.type_map)
    # print(visitor.method_map)
    # In ra danh sách các cạnh dưới dạng tuple: (parent, edge_label, child)
    for edge in visitor.edges:
        logging.info(edge)
    
    logging.info(jsonl_dataset['func'][300])
    # Ví dụ in ra các node với subindex
    # print("\nCác node đã duyệt:")
    # for node in visitor.node_mapping.values():
    #     with open('test.txt','a') as f:
    #         f.write(str(node.node)+'\n')
    #         f.write('-'*50+'\n')
        # print(node.node)
        # print('-'*50)